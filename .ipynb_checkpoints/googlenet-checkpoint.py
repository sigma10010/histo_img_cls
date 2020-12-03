'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        '''
        output: cat n1x1, n3x3, n5x5, pool_planes
        '''
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)
    
class InceptionEmbeddingBreakHis(nn.Module):
    def __init__(self, n_classes):
        super(InceptionEmbeddingBreakHis, self).__init__()
        self.n_classes = n_classes
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=7, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
            nn.Conv2d(4, 4, kernel_size=1, padding=1),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        self.a3 = Inception(8,  4,  6, 8, 8, 2, 2)
        self.b3 = Inception(16, 8, 8, 16, 16, 4, 4)

        self.maxpool = nn.MaxPool2d((3,5), stride=2, padding=1)

        self.a4 = Inception(32, 8, 8, 16, 16, 4, 4)
        self.b4 = Inception(32, 8, 8, 16, 16, 4, 4)
#         self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
#         self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
#         self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(32, 16, 16, 32, 32, 8, 8)
        self.b5 = Inception(64, 16, 16, 32, 32, 8, 8)
        
        self.a6 = Inception(64, 16, 16, 32, 32, 8, 8)
        self.b6 = Inception(64, 32, 32, 64, 64, 16, 16)
        
        self.a7 = Inception(128, 32, 32, 64, 64, 16, 16)
        self.b7 = Inception(128, 32, 32, 64, 64, 16, 16)

        self.avgpool = nn.AvgPool2d((8,10), stride=1)
        self.linear = nn.Linear(128, self.n_classes+10) # dimension reduction
        
        

    def forward(self, x):
#         print('input size: ', x.size())
        out = self.pre_layers(x)
#         print('pre_layers size: ', out.size())
        out = self.a3(out)
#         print('a3 size: ', out.size())
        out = self.b3(out)
#         print('b3 size: ', out.size())
        out = self.maxpool(out)
        out = self.a4(out)
#         print('a4 size: ', out.size())
        out = self.b4(out)
#         print('b4 size: ', out.size())
#         out = self.c4(out)
#         print('c4 size: ', out.size())
#         out = self.d4(out)
#         print('d4 size: ', out.size())
#         out = self.e4(out)
#         print('e4 size: ', out.size())
        out = self.maxpool(out)
        out = self.a5(out)
#         print('a5 size: ', out.size())
        out = self.b5(out)
#         print('b5 size: ', out.size())
        out = self.maxpool(out)
        
        out = self.a6(out)
#         print('a6 size: ', out.size())
        out = self.b6(out)
#         print('b6 size: ', out.size())

        out = self.maxpool(out)
        out = self.a7(out)
        out = self.b7(out)
#         print('b7 size: ', out.size())
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
#         print('view size: ', out.size())
        out = self.linear(out)
#         print('linear size: ', out.size())
        return out

    def get_embedding(self, x):
        return self.forward(x)
    
class InceptionEmbedding2241D(nn.Module):
    def __init__(self, n_classes):
        super(InceptionEmbedding2241D, self).__init__()
        self.n_classes = n_classes
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=1, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 256, 160, 320, 32, 128, 128)
        
        self.a6 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b6 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, self.n_classes+10) # dimension reduction
        
        

    def forward(self, x):
#         print('input size: ', x.size())
        out = self.pre_layers(x)
#         print('pre_layers size: ', out.size())
        out = self.a3(out)
#         print('a3 size: ', out.size())
        out = self.b3(out)
#         print('b3 size: ', out.size())
        out = self.maxpool(out)
        out = self.a4(out)
#         print('a4 size: ', out.size())
        out = self.b4(out)
#         print('b4 size: ', out.size())
        out = self.c4(out)
#         print('c4 size: ', out.size())
        out = self.d4(out)
#         print('d4 size: ', out.size())
        out = self.e4(out)
#         print('e4 size: ', out.size())
        out = self.maxpool(out)
        out = self.a5(out)
#         print('a5 size: ', out.size())
        out = self.b5(out)
#         print('b5 size: ', out.size())
        out = self.maxpool(out)
        
        out = self.a6(out)
#         print('a6 size: ', out.size())
        out = self.b6(out)
#         print('b6 size: ', out.size())
               
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
#         print('view size: ', out.size())
        out = self.linear(out)
#         print('linear size: ', out.size())
        return out

    def get_embedding(self, x):
        return self.forward(x)
    
class InceptionEmbedding1121D(nn.Module):
    def __init__(self, n_classes):
        super(InceptionEmbedding1121D, self).__init__()
        self.n_classes = n_classes
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=1, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

#         self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.b5 = Inception(832, 256, 160, 320, 32, 128, 128)
        
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, self.n_classes+10) # dimension reduction
        
        

    def forward(self, x):
#         print('input size: ', x.size())
        out = self.pre_layers(x)
#         print('pre_layers size: ', out.size())
        out = self.a3(out)
#         print('a3 size: ', out.size())
        out = self.b3(out)
#         print('b3 size: ', out.size())
        out = self.maxpool(out)
        out = self.a4(out)
#         print('a4 size: ', out.size())
        out = self.b4(out)
#         print('b4 size: ', out.size())
        out = self.c4(out)
#         print('c4 size: ', out.size())
        out = self.d4(out)
#         print('d4 size: ', out.size())
        out = self.e4(out)
#         print('e4 size: ', out.size())
        out = self.maxpool(out)
        out = self.a5(out)
#         print('a5 size: ', out.size())
        out = self.b5(out)
#         print('b5 size: ', out.size())
               
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
#         print('view size: ', out.size())
        out = self.linear(out)
#         print('linear size: ', out.size())
        return out

    def get_embedding(self, x):
        return self.forward(x)
    
class InceptionEmbadding2D(nn.Module):
    def __init__(self):
        super(InceptionEmbadding2D, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=1, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256,  64,  96, 128, 16, 32, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(256,  64,  96, 128, 16, 32, 32)
        self.b4 = Inception(256,  64,  96, 128, 16, 32, 32)
        self.c4 = Inception(256,  64,  96, 128, 16, 32, 32)
        self.d4 = Inception(256,  64,  96, 128, 16, 32, 32)
        self.e4 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.a5 = Inception(480, 128, 128, 192, 32, 96, 64)
        self.b5 = Inception(480,  64,  96, 128, 16, 32, 32)
        
        self.a6 = Inception(256,  32,  96, 64, 16, 16, 16)
        self.b6 = Inception(128, 32, 192, 32, 48, 16, 16)

#         self.avgpool = nn.AvgPool2d(7, stride=1)
        
        

    def forward(self, x):
#         print('input size: ', x.size())
        out = self.pre_layers(x)
#         print('pre_layers size: ', out.size())
        out = self.a3(out)
#         print('a3 size: ', out.size())
        out = self.b3(out)
#         print('b3 size: ', out.size())
        out = self.maxpool(out)
        out = self.a4(out)
#         print('a4 size: ', out.size())
        out = self.b4(out)
#         print('b4 size: ', out.size())
        out = self.c4(out)
#         print('c4 size: ', out.size())
        out = self.d4(out)
#         print('d4 size: ', out.size())
        out = self.e4(out)
#         print('e4 size: ', out.size())
        out = self.maxpool(out)
        out = self.a5(out)
#         print('a5 size: ', out.size())
        out = self.b5(out)
#         print('b5 size: ', out.size())
        out = self.maxpool(out)
        
        out = self.a6(out)
#         print('a6 size: ', out.size())
        out = self.b6(out)
#         print('b6 size: ', out.size())
               
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         print('view size: ', out.size())
#         out = self.linear(out)
#         print('linear size: ', out.size())
        return out

    def get_embedding(self, x):
        return self.forward(x)
    

class GoogLeNet2D(nn.Module):
    # work with InceptionEmbadding2D
    def __init__(self, embedding_net, n_classes):
        super(GoogLeNet2D, self).__init__()
        self.n_classes=n_classes
        self.embedding=embedding_net
        self.nonlinear = nn.PReLU()
        
        

    def forward(self, x):
        out = self.embedding(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
#         print('view size: ', out.size())
        out = self.nonlinear(out)
        scores = F.log_softmax(self.linear(out), dim=-1)
        
#         print('linear size: ', scores.size())
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding(x))
    
class GoogLeNet1D(nn.Module):
    # work with InceptionEmbadding1D
    def __init__(self, embedding_net, n_classes):
        super(GoogLeNet1D, self).__init__()
        self.embedding=embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(self.n_classes+10, self.n_classes) # linear classifier
        

    def forward(self, x):
        out = self.embedding(x)
        out = self.nonlinear(out)
        scores = F.log_softmax(self.fc(out), dim=-1)
        
#         print('size: ', scores.size())
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding(x))
    
    def get_classification(self, x):
        return self.forward(x)

def test():
    embedding=InceptionEmbedding2241D(2)
#     net=embadding
    net = GoogLeNet1D(embedding, 2)
#     x = torch.randn(8,3,112,112)
    x = torch.randn(8,3,224,224)
    y = net(x)
    print(net)

# test()
