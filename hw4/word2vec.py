import torch
import random
from collections import Counter
import argparse
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.util import preprocess, create_context_target, convert_one_hot
from w2vec.CBow import CustomCBOW
from w2vec.SkipGram import CustomSkipGram
import pickle # for save
from common.util import most_similar_byEmb

def word2vec_trainer(corpus, word2ind, id2word, mode="CBOW", dimension=64, learning_rate=0.01, iteration=50000, batch_size=500, window_size=3):
    vocab_size = len(word2ind)
    losses = []
    sum_iter = 0
    parallel_num = 2 * window_size # number of parallel affine layers
    slice_len = 100
    #################### model initialization ####################
    if mode == "CBOW":
        model = CustomCBOW(vocab_size, dimension, vocab_size, parallel_num)
    elif mode == "SG":
        model = CustomSkipGram(vocab_size, dimension, vocab_size, parallel_num)
    else:
        print("Unkwnown mode : " + mode)
        exit()
    optimizer = SGD(lr=learning_rate)
    ######## corpus slicing (to avoid memory excess error) ########
    corpus_sliced = []
    head = 0
    tail = 0
    while tail < len(corpus):
        tail = min(tail + slice_len, len(corpus))
        corpus_sliced.append(corpus[max(0, head - 2 * window_size) : tail])
        head = tail
    print("Length of one slice : %d / Number of slices : %d" % (slice_len, len(corpus_sliced)))
    ######################## Learning ########################
    for i, sub_corpus in enumerate(corpus_sliced):
        print("Learning for corpus slice #%d" % (i+1))
        contexts, target = create_context_target(sub_corpus, window_size)
        target = convert_one_hot(target, vocab_size).float()
        contexts = convert_one_hot(contexts, vocab_size).float()
        batch_size = min(batch_size, len(target))
        for i in range(iteration+1):
            sum_iter += 1
            ################## getRandomContext ##################
            index = torch.randperm(len(target))[0:batch_size]
            centerWord, contextWords = target[index], contexts[index]
            ################## learning ##################    
            loss = model.forward(contextWords, centerWord)
            model.backward()
            optimizer.update(model)
            losses.append(loss)
            ################## learning rate decay ##################
            lr = learning_rate*(1-i/iteration)
            optimizer.set_lr(lr)
            #########################################################
            if sum_iter%5==0:
                avg_loss=sum(losses)/len(losses)
                print("Iteration : %d / Loss : %f" %(20*sum_iter, avg_loss))
                w_emb, _ = model.get_inputw()
                # saved
                params = {}
                params['word_vecs'] = w_emb
                #params['word_out'] = W_out
                params['word2id'] = word2ind
                params['id2word'] = id2word
                if mode == "CBOW":
                    filename = 'cbowpkl_file'
                elif mode == "SG":
                    filename = 'sgpkl_file'
                with open(filename, 'wb') as f:
                    pickle.dump(params, f, protocol=3)


    ######### Extract W matrix #########
    W_in, b_in = model.get_inputw()
    W_out, b_out = model.get_outputw()
    W_emb = W_in
    ####################################
    return W_emb, W_out

def main():
    mode = "SG"
    with open('text8', 'r') as f:
        text = f.read()
	# Write your code of data processing, training, and evaluation
	# Full training takes very long time. We recommend using a subset of text8 when you debug
    corpus, word2id, id2word = preprocess(text, subset=1)
    print("Processing completed")
    W_emb, W_out = word2vec_trainer(corpus, word2id, id2word, mode=mode, learning_rate=0.20, iteration=5, window_size=1)

    # saved
    params = {}
    params['word_vecs'] = W_emb
    params['word_out'] = W_out
    params['word2id'] = word2id
    params['id2word'] = id2word
    if mode == "CBOW":
        filename = 'cbowpkl_file'
    elif mode == "SG":
        filename = 'sgpkl_file'
    with open(filename, 'wb') as f:
        pickle.dump(params, f, protocol=3)
    
main()
