import torch
import torch.nn as nn


class MLP_model(nn.Module):
    def __init__(self):
        super().__init__()

        ##############################################################################################################
        #                         TODO : MLP 모델 생성 (구조는 실험해 보면서 결과가 좋은 것으로 사용할 것)                 #
        ##############################################################################################################
        H1 = 200
        H2 = 100
        H3 = 50
        keep_rate = 0.5
        
        self.lin1 = nn.Linear(3*32*32, H1)
        self.lin2 = nn.Linear(H1, H2)
        self.lin3 = nn.Linear(H2, H3)
        self.lin4 = nn.Linear(H3, 10)

        torch.nn.init.xavier_normal(self.lin1.weight)
        torch.nn.init.xavier_normal(self.lin2.weight)
        torch.nn.init.xavier_normal(self.lin3.weight)
        torch.nn.init.xavier_normal(self.lin4.weight)

        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)
        self.bn3 = nn.BatchNorm1d(H3)

        #self.dr1 = torch.nn.Dropout(1 - keep_rate)
        #self.dr2 = torch.nn.Dropout(1 - keep_rate)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        x = torch.sigmoid(self.bn1(self.lin1(x)))
        #x = self.dr1(x)
        x = torch.sigmoid(self.bn2(self.lin2(x)))
        #x = self.dr2(x)
        x = torch.sigmoid(self.bn3(self.lin3(x)))
        
        log_softmax = torch.nn.LogSoftmax(dim=1)
        x = log_softmax(self.lin4(x))
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
        return x


