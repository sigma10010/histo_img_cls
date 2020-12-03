import os
import torch
from torchvision import transforms
from datasets import BreakHis, BreakHisPIL
from networks import TripletClassificationNet, BreakHisEmbeddingNet
from googlenet import InceptionEmbeddingBreakHis
from resnet import ResNet18

from evaluation import *
from save_load import *

from losses import TripletClassificationLoss
from metrics import TripletAccumulatedAccuracyMetric


def run_eval_net(ckpt, ckpt_w, magnification = None):
    fold = int(ckpt.split('_')[4][-1])
    if magnification is not None:
        ptest = '/mnt/DATA_OTHER/breakHis/mkfold/fold%d/test/%dX/'%(fold,magnification)
    else:
        ptest = '/mnt/DATA_OTHER/breakHis/mkfold/fold%d/test/'%fold
    test_dataset = BreakHisPIL(ptest,data_aug=False, transform=transforms.Compose([
                                     transforms.ToTensor()

                                 ]))
    n_classes = len(test_dataset.labels_set)
    
    cuda = torch.cuda.is_available()
    batch_size = 200
    kwargs = {'num_workers': 3, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    margin = 1
    embedding_net = BreakHisEmbeddingNet(n_classes)
    model = TripletClassificationNet(embedding_net, n_classes)
    if cuda:
        model.cuda()
    model.eval()
    model=load_model_ckpt_w(model, ckpt_w)
    
    ps_m, miscls = patient_score_matrix(model, test_loader, cuda)
    ps = (ps_m[:,0]/ps_m[:,1]).mean()
    
    c = conf_matrix_breakHis(model,test_loader,list(test_loader.dataset.labels_set), cuda)
    f1 = f1_score(c)
    acc = accuracy(c)
    return ps, ps_m, acc, f1, c, miscls

if __name__ == "__main__":
#     save_fold = '/mnt/DATA_OTHER/breakHis/results/evaluation'
#     ckpts = []
#     for _, _, files in os.walk('./checkpoint/'):
#         for x in files:
#             if x.endswith(('.t7')):
#                 ckpts.append(x)
#     result_str = ''     
#     for ckpt in ckpts:
#         print(ckpt)
#         result_str += ckpt + ':\n'
#         for magnification in [40, 100, 200, 400]:
#             a = os.path.join(save_fold, ckpt.split('.')[0], str(magnification))
#             if not os.path.exists(a):
#                 os.makedirs(a)
                
#             result_str += (' ' + str(magnification) + 'X' + ':\n')
#             ps, ps_m, acc, f1, c = run_eval_net(ckpt, magnification)
            
#             np.save(os.path.join(a, 'patient_matrix.npy'), np.array(ps_m))
#             np.save(os.path.join(a, 'confusion_matrix.npy'), np.array(c))
            
#             result_str += "pateient score:" + str(ps) + '\n'
#             result_str += "accuracy:" + str(acc) + '\n'
#             result_str += "f1-score:[%f, %f]"%(f1[0], f1[1]) + '\n'
            
# #             with open("test.txt", "w") as fid:
# #                 fid.write(result_str)
            
#     with open('results/' + "breakHis_mag.txt", "w") as fid:
#         fid.write(result_str)

    save_fold = '/mnt/DATA_OTHER/breakHis/results/evaluation'
    ckpts_fold = '/mnt/DATA_OTHER/breakHis/results/checkpoint/breakHis/'
    ckpts = []
    ckpts_w = []
    for _, _, files in os.walk(ckpts_fold):
        for x in files:
            if x.endswith('.t7') and x.split('_')[-2]=='2l1**1.5beta':
                ckpts.append(x)
                ckpts_w.append(os.path.join(ckpts_fold,x))
    result_str = ''     
    for i, ckpt in enumerate(ckpts):
        print(ckpt)
        result_str += ckpt + ':\n'
        a = os.path.join(save_fold, ckpt)
        if not os.path.exists(a):
            os.makedirs(a)
                
        ps, ps_m, acc, f1, c, miscls = run_eval_net(ckpt, ckpts_w[i])
            
        np.save(os.path.join(a, 'patient_matrix.npy'), np.array(ps_m))
        np.save(os.path.join(a, 'confusion_matrix.npy'), np.array(c))
        np.save(os.path.join(a, 'miscls.npy'), np.array(miscls))
            
        result_str += "pateient score:" + str(ps) + '\n'
        result_str += "accuracy:" + str(acc) + '\n'
        result_str += "f1-score:[%f, %f]"%(f1[0], f1[1]) + '\n'
            
    with open('results/' + "breakHis_optimal.txt", "w") as fid:
        fid.write(result_str)