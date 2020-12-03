import os
import torch
from torchvision import transforms
from datasets import BaseHistoFromSampler
from networks import EmbeddingNet, ClassificationNet
from googlenet import InceptionEmbedding2241D
from resnet import ResNet18

from evaluation import *
from save_load import *

def run_eval_net(ckpt):
    result_fold = '/mnt/DATA_OTHER/liverC/results'
    experiment = 'experiments'
    number = '20201113-500'

    val_data = np.load(os.path.join(result_fold, experiment, number, 'val_data.npy'))
    val_dataset_w = np.load(os.path.join(result_fold, experiment, number, 'val_dataset_w.npy'))

    test_dataset = BaseHistoFromSampler(imgs=val_data, imgs_w=val_dataset_w, size=224, transform=transforms.Compose([
                                 transforms.ToTensor()
                                 
                             ])
                            )
    n_classes = len(test_dataset.labels_set)
    
    cuda = torch.cuda.is_available()
    batch_size = 200
    kwargs = {'num_workers': 3, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    margin = 1
    if ckpt.split('_')[1]=='pnn2':
        embedding_net = EmbeddingNet(n_classes)
    elif ckpt.split('_')[1]=='resnet':
        embedding_net = ResNet18()
    elif ckpt.split('_')[1]=='googlenet':
        embedding_net = InceptionEmbedding2241D(n_classes)
    else:
        raise NameError('embeddingnet options: pnn2, resnet, googlenet')
    model = ClassificationNet(embedding_net, n_classes)
    if cuda:
        model.cuda()
    model.eval()
    model=load_model_ckpt(model, ckpt)
    
    c = conf_matrix(model,test_loader,list(test_loader.dataset.labels_set), cuda)
    f1 = f1_score(c)
    acc = accuracy(c)
    return acc, f1

if __name__ == "__main__":
    
    ckpts = []
    for _, _, files in os.walk('./checkpoint/'):
        for x in files:
            if x.endswith('.t8') or x.endswith('.t0'):
                ckpts.append(x)
    result_str = ''     
    for ckpt in ckpts:
        print(ckpt)
        result_str += ckpt + ':\n'
        
        acc, f1 = run_eval_net(ckpt)
            
        result_str += "accuracy:" + str(acc) + '\n'
        result_str += "f1-score:[%f, %f]"%(f1[0], f1[1]) + '\n'
            
    with open('results/' + "liverC_g500.txt", "w") as fid:
        fid.write(result_str)