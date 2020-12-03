from torchvision import transforms
# from datasets import BaseHistoFromSampler, SlideDataSampler, SlideSampler
from datasets import BreakHis, BreakHisPIL

# ========================liver Dataset===================================================================================
# ptrain='/mnt/DATA_OTHER/liverC/patches/train/'
# pval='/mnt/DATA_OTHER/liverC/patches/validation/'

# train_slide_S = SlideSampler(root_dir=ptrain, n_sample=30)
# _, train_slides = train_slide_S._sampler()

# val_slide_S = SlideSampler(root_dir=pval, n_sample=20)
# _, val_slides = val_slide_S._sampler()

# train_slide_SD = SlideDataSampler(slide_dirs=train_slides,n_sample=500)
# train_data, train_dataset_w = train_slide_SD._sampler()

# val_slide_SD = SlideDataSampler(slide_dirs=val_slides,n_sample=500)
# val_data, val_dataset_w = val_slide_SD._sampler()

# train_dataset = BaseHistoFromSampler(imgs=train_data, imgs_w=train_dataset_w, size=224, transform=transforms.Compose([
#                                  transforms.ToTensor()
                                 
#                              ]))
# test_dataset = BaseHistoFromSampler(imgs=val_data, imgs_w=val_dataset_w, size=224, transform=transforms.Compose([
#                                  transforms.ToTensor()
                                 
#                              ])
#                             )
# n_classes = len(train_dataset.labels_set)

# ========================ring cell Dataset===================================================================================
# ptrain='/mnt/DATA_OTHER/digestPath/Signet_ring_cell_dataset/sig-train-pos/patches/ringcell/train/'
# pval='/mnt/DATA_OTHER/digestPath/Signet_ring_cell_dataset/sig-train-pos/patches/ringcell/validation/'

# train_slide_S = SlideSampler(root_dir=ptrain, n_sample=300)
# _, train_slides = train_slide_S._sampler()

# val_slide_S = SlideSampler(root_dir=pval, n_sample=200)
# _, val_slides = val_slide_S._sampler()

# train_slide_SD = SlideDataSampler(slide_dirs=train_slides,n_sample=500)
# train_data, train_dataset_w = train_slide_SD._sampler()

# val_slide_SD = SlideDataSampler(slide_dirs=val_slides,n_sample=500)
# val_data, val_dataset_w = val_slide_SD._sampler()

# train_dataset = BaseHistoFromSampler(imgs=train_data, imgs_w=train_dataset_w, size=112, transform=transforms.Compose([
#                                  transforms.ToTensor()
                                 
#                              ]))
# test_dataset = BaseHistoFromSampler(imgs=val_data, imgs_w=val_dataset_w, size=112, transform=transforms.Compose([
#                                  transforms.ToTensor()
                                 
#                              ])
#                             )
# n_classes = len(train_dataset.labels_set)

# =========================BreakHis============================================================
fold = 5
ptrain = '/mnt/DATA_OTHER/breakHis/mkfold/fold%d/train/'%fold
ptest = '/mnt/DATA_OTHER/breakHis/mkfold/fold%d/test/'%fold
train_dataset = BreakHisPIL(ptrain, transform=transforms.Compose([
                                transforms.ToTensor()
                                 
                             ]))
test_dataset = BreakHisPIL(ptest, data_aug = False, transform=transforms.Compose([
                                transforms.ToTensor()
                                 
                             ]))
n_classes = len(train_dataset.labels_set)

rate_b2all = train_dataset.statistic()['B']/(train_dataset.statistic()['B'] + train_dataset.statistic()['M'])
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
from datasets import TripletClsBreakHisPIL, SiameseClsBreakHisPIL
# !!!!!!!!!!note the defualt size of dataset!!!!!!!!!!!!
triplet_train_dataset = TripletClsBreakHisPIL(train_dataset, train=True) # Returns pairs of images and target same/different
triplet_test_dataset = TripletClsBreakHisPIL(test_dataset, train=False)
# pair_train_dataset = SiameseClsBreakHisPIL(train_dataset, train=True) # Returns pairs of images and target same/different
# pair_test_dataset = SiameseClsBreakHisPIL(test_dataset, train=False)

batch_size = 50
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


from networks import TripletClassificationNet, BreakHisEmbeddingNet, SiameseClassificationNet
# from googlenet import InceptionEmbedding2241D

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
from networks import BreakHisEmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric
embedding_net = BreakHisEmbeddingNet(n_classes)
model = TripletClassificationNet(embedding_net, n_classes)
model_name = 'baseline_triplet_c_breakHis5_daug_2l1**1.2beta'

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
    
# loss_fn = TripletClassificationLoss(margin, torch.nn.NLLLoss())
# loss_fn = TripletClassificationLoss(margin, FocalLoss(gamma = 0, alpha = balance))
loss_fn = TripletClassificationLoss(margin, GeneralizedL1Loss(alpha = 2, gamma = 1.2, beta = balance))
# loss_fn = ContrastiveClassificationLoss(margin, GeneralizedL1Loss(alpha = 2, gamma = 1.5, beta = balance))
# loss_fn = GeneralizedL1Loss(alpha = 2, gamma = 1.5, beta = balance)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 40
log_interval = 30


fit(triplet_train_loader, triplet_test_loader, model, model_name, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[TripletAccumulatedAccuracyMetric()])
# fit(pair_train_loader, pair_test_loader, model, model_name, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[SiameseAccumulatedAccuracyMetric()])
# fit(train_loader, test_loader, model, model_name, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
