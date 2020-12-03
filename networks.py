import torch.nn as nn
import torch.nn.functional as F

class BreakHisEmbeddingNet(nn.Module):
    '''
    input size 3*460*700
    '''
    def __init__(self, n_classes):
        super(BreakHisEmbeddingNet, self).__init__()
        self.n_classes = n_classes
        self.convnet = nn.Sequential(nn.Conv2d(3, 4, kernel_size=(3, 5)), 
                        nn.PReLU(),
                        nn.MaxPool2d((2,3), stride=2),
                        nn.Conv2d(4, 4, kernel_size=(3, 5)), 
                        nn.PReLU(),
                        nn.MaxPool2d((2,3), stride=2),
                        nn.Conv2d(4, 8, kernel_size=(3, 5)), 
                        nn.PReLU(),
                        nn.MaxPool2d((2,3), stride=2),
                        nn.Conv2d(8, 16, kernel_size=(3, 5)), 
                        nn.PReLU(),
                        nn.MaxPool2d((2,3), stride=2),
                        nn.Conv2d(16, 32, kernel_size=(3, 5)), 
                        nn.PReLU(),
                        nn.MaxPool2d((2,3), stride=2),
                        nn.Conv2d(32, 64, kernel_size=(3, 5)), 
                        nn.PReLU(),
                        nn.MaxPool2d((2,3), stride=2)
                        )

        self.fc = nn.Sequential(
                                nn.Linear(64 * 5 * 6, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, self.n_classes + 10)
                                )

    def forward(self, x):
        output = self.convnet(x)
#         print(output.shape)
        output = output.view(output.size()[0], -1)
#         print(output.shape)
        output = self.fc(output)
#         print(output.shape)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet(nn.Module):
    '''
    for liveC
    input size 3*224*224
    '''
    def __init__(self, n_classes):
        super(EmbeddingNet, self).__init__()
        self.n_classes = n_classes
        self.convnet = nn.Sequential(nn.Conv2d(3, 8, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(8, 16, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(16, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2)
                                    )

        self.fc = nn.Sequential(nn.Linear(64 * 10 * 10, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, self.n_classes + 10)
                                )

    def forward(self, x):
        output = self.convnet(x)
#         print(output.shape)
        output = output.view(output.size()[0], -1)
#         print(output.shape)
        output = self.fc(output)
#         print(output.shape)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
class EmbeddingNet2D(nn.Module):
    def __init__(self, n_classes):
        super(EmbeddingNet2D, self).__init__()
        self.n_classes = n_classes
        self.convnet = nn.Sequential(nn.Conv2d(3, 8, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(8, 16, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(16, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64,32,5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, self.n_classes, 3)
                                    )


    def forward(self, x):
        output = self.convnet(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(self.n_classes+10, self.n_classes) # linear classifier

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.softmax(self.fc(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
    
    def get_classification(self, x):
        return self.forward(x)
    
class FullyConvClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(FullyConvClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(1024, n_classes+10)

    def forward(self, x):
        output = self.embedding_net(x)
        output = output.view(output.size(0), -1)
        output = self.nonlinear(output)
        scores = F.softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        output = self.embedding_net(x)
        output = output.view(output.size(0), -1)
        return self.fc1(self.nonlinear(output))
    
    def get_classification(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class SiameseClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(SiameseClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(self.n_classes+10, self.n_classes) # linear classifier
        

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        c_output1 = self.nonlinear(output1)
        c_output1 = F.softmax(self.fc(c_output1), dim=-1)
        c_output2 = self.nonlinear(output2)
        c_output2 = F.softmax(self.fc(c_output2), dim=-1)
        return output1, output2, c_output1

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_classification(self, x):
        out=self.embedding_net(x)
        out=self.nonlinear(out)
        return F.softmax(self.fc(out), dim=-1)
    
class SiameseFullyConvClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(SiameseFullyConvClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(self.n_classes+10, self.n_classes) # linear classifier
        

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        c_output1 = self.nonlinear(output1)
        c_output1 = F.softmax(self.fc(c_output1), dim=-1)
        c_output2 = self.nonlinear(output2)
        c_output2 = F.softmax(self.fc(c_output2), dim=-1)
        return output1, output2, c_output1

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_classification(self, x):
        out=self.embedding_net(x)
        out=self.nonlinear(out)
        return F.softmax(out, dim=-1)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(TripletClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(self.n_classes+10, self.n_classes) # linear classifier
        

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        c_output1 = self.nonlinear(output1)
        c_output1 = F.softmax(self.fc(c_output1), dim=-1)
        
        return output1, output2, output3, c_output1

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_classification(self, x):
        out=self.embedding_net(x)
        out=self.nonlinear(out)
        return F.softmax(self.fc(out), dim=-1)