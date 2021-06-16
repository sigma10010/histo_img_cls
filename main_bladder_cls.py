from torchvision import transforms
# from datasets import BaseHistoFromSampler, SlideDataSampler, SlideSampler
from datasets import BladderPIL

ptrain = '/mnt/DATA_OTHER/bladder/images/train/'
ptest = '/mnt/DATA_OTHER/bladder/images/valid/'

train_dataset = BladderPIL(ptrain, transform=transforms.Compose([
                                transforms.ToTensor()
                                 
                             ]))
test_dataset = BladderPIL(ptest, data_aug = False, transform=transforms.Compose([
                                transforms.ToTensor()
                                 
                             ]))
n_classes = len(train_dataset.labels_set)
rate_b2all = train_dataset.statistic()['0']/(train_dataset.statistic()['0'] + train_dataset.statistic()['1'])
balance = 1- rate_b2all

# ============================================================================================================
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

# ============================================================================================================

# Set up data loaders
from datasets import TripletClsPIL, SiameseClsPIL
# !!!!!!!!!!note the defualt size of dataset!!!!!!!!!!!!
triplet_train_dataset = TripletClsPIL(train_dataset, train=True) # Returns pairs of images and target same/different
triplet_test_dataset = TripletClsPIL(test_dataset, train=False)
# pair_train_dataset = SiameseClsPIL(train_dataset, train=True) # Returns pairs of images and target same/different
# pair_test_dataset = SiameseClsPIL(test_dataset, train=False)

batch_size = 100
kwargs = {'num_workers': 3, 'pin_memory': True} if cuda else {}

triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# pair_train_loader = torch.utils.data.DataLoader(pair_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# pair_test_loader = torch.utils.data.DataLoader(pair_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up data loaders baseline
# batch_size = 56
# kwargs = {'num_workers': 3, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)



# Set up the network and training parameters 

# ========================liver Dataset===================================================================================


from networks import TripletClassificationNet, EmbeddingNet, SiameseClassificationNet
from googlenet import InceptionEmbedding2241D
from resnet import ResNet18

from losses import TripletClassificationLoss, ContrastiveClassificationLoss, FocalLoss, GeneralizedL1Loss
from metrics import TripletAccumulatedAccuracyMetric, SiameseAccumulatedAccuracyMetric

margin = 1.0
# # ------------------------- googlenet ----------------------------------------------------------------------
# # embedding_net = InceptionEmbedding2241D(n_classes)
# # model_name = 'Inception_Triplet_C_liverC'
# # ------------------------- normal CNN ----------------------------------------------------------------------
# embedding_net = BreakHisEmbeddingNet(n_classes)
# model_name = 'baseline_Triplet_C_breakHis4_daug_l1**1.5'

# model = TripletClassificationNet(embedding_net, n_classes)
# print('==================================', model)

# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric
# embedding_net = InceptionEmbedding2241D(n_classes)
embedding_net = EmbeddingNet(n_classes)
model = TripletClassificationNet(embedding_net, n_classes)
model_name = 'pnn2_triplet_c_bladder_2l1**15'

# model = SiameseClassificationNet(embedding_net, n_classes)
# model_name = 'baseline_pair_c_breakHis5_daug_2l1**1.5beta_m20'

# ========================ring cell Dataset===================================================================================
# from networks import TripletClassificationNet
# from googlenet import InceptionEmbedding1121D
# from losses import TripletClassificationLoss
# from metrics import TripletAccumulatedAccuracyMetric

# margin = 1
# embedding_net = InceptionEmbedding1121D(n_classes)
# model = TripletClassificationNet(embedding_net, n_classes)
# model_name = 'Inception_Triplet_ring'

# ============================================================================================================

if cuda:
    model.cuda()
    
# loss_fn = FocalLoss(gamma = 0)
# loss_fn = ContrastiveClassificationLoss(margin, FocalLoss(gamma = 0))
# loss_fn = TripletClassificationLoss(margin, GeneralizedL1Loss(alpha = 2, gamma = 1.2, beta = balance))
loss_fn = TripletClassificationLoss(margin, GeneralizedL1Loss(alpha = 2, gamma = 1.5, beta = balance))
# loss_fn = GeneralizedL1Loss(alpha = 2, gamma = 1.5, beta = balance)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 400
log_interval = 10


fit(triplet_train_loader, triplet_test_loader, model, model_name, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[TripletAccumulatedAccuracyMetric()])
# fit(pair_train_loader, pair_test_loader, model, model_name, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[SiameseAccumulatedAccuracyMetric()])
# fit(train_loader, test_loader, model, model_name, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
