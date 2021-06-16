import numpy as np
from PIL import Image
from skimage import color, io
from skimage.transform import resize

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

from xml.etree import ElementTree
import csv

import os
import torch
from torchvision import transforms
from transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, Compose

import shutil
import random

class divide_data():
    '''
    divide data by slide level
    root_dir: parent root of all/
    rate: train:val:test
    '''
    def __init__(self, root_dir, rate=[1,1,1]):
        self.rate = rate  # train:val:test
        self.root = root_dir
        self.root_train = self.root+'train/'
        self.root_test = self.root+'test/'
        self.root_val = self.root+'validation/'
        self.root_all = self.root+'all/'
        if not os.path.exists(self.root_train):
            os.mkdir(self.root_train)
        if not os.path.exists(self.root_test):
            os.mkdir(self.root_test)
        if not os.path.exists(self.root_val):
            os.mkdir(self.root_val)

    def reset(self):
        for f in os.listdir(self.root_train):
            shutil.move(self.root_train+f,self.root_all)
        for f in os.listdir(self.root_test):
            shutil.move(self.root_test+f,self.root_all)
        for f in os.listdir(self.root_val):
            shutil.move(self.root_val+f,self.root_all)
        return 'sucess'

    def divide(self):   
        index=list(range(len(os.listdir(self.root_all))))
        random.shuffle(index)
        dirs=[]
        for _, f in enumerate(os.listdir(self.root_all)):
            dirs.append(f)
        for i in range(len(dirs)):
            if i<len(dirs)*(self.rate[0]/(np.array(self.rate).sum())):
                shutil.move(self.root_all+dirs[index[i]],self.root_train)
            elif len(dirs)*(self.rate[0]/np.array(self.rate).sum())<=i<len(dirs)-len(dirs)*(self.rate[2]/(np.array(self.rate).sum())):
                shutil.move(self.root_all+dirs[index[i]],self.root_val)
            else:
                shutil.move(self.root_all+dirs[index[i]],self.root_test)
        return 'sucess'

    
class SlideDataSampler():
    '''
    data organization: root/slide/patch/class
    sample
    '''
    def __init__(self, slide_dirs, n_sample=500, label2rm=['U',]):
        self.n_sample = n_sample # number of sample per class per slide
        self.roots = slide_dirs
        self.l2rm = label2rm
            
    def _sampler(self):
        dataset = []
        dataset_w = []
        for root in self.roots:
            imglist = []
            wholepath = []
            for label in os.listdir(root):
                # label to keep out
                if label in self.l2rm:
                    continue
                img_list = [f for f in os.listdir(os.path.join(root, label))]
                img_list = np.array(img_list)
                if len(img_list)>self.n_sample:
                    chosen_img = np.random.choice(img_list, size=self.n_sample, replace=False)
                    chosen_whole_path = [os.path.join(root, label, f) for f in  chosen_img]
                else:
                    chosen_img = img_list
                    chosen_whole_path = [os.path.join(root, label, f) for f in  chosen_img]
                imglist.append(list(chosen_img))
                wholepath.append(list(chosen_whole_path))
            dataset.append(imglist)
            dataset_w.append(wholepath)
            
        # 3d to 1d
        imglist1d = [z for x in dataset for y in x for z in y]
        img_w_list1d = [z for x in dataset_w for y in x for z in y]
        
        return np.asarray(imglist1d), np.asarray(img_w_list1d)
    
class SlideSampler():
    '''
    data organization: root/slide/
    sample
    '''
    def __init__(self, root_dir, n_sample=5):
        self.n_sample = n_sample # number of sample per class per slide
        self.root = root_dir
    
    def _sampler(self):
        s_list = [slide for slide in os.listdir(self.root)]
        if len(s_list)>self.n_sample:
            chosen_s = np.random.choice(s_list, size=self.n_sample, replace=False)
            chosen_whole_path = [os.path.join(self.root, f) for f in  chosen_s]
        else:
            chosen_s = s_list
            chosen_whole_path = [os.path.join(self.root, f) for f in  chosen_s]
        return chosen_s, chosen_whole_path        


class BaseHistoNew(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """

    def __init__(self, root_dir, size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.size = size
        self.transform = transform
        self.label_name_list =list(sorted(set(self.walk_root_dir()[2]))) # sorted to keep consistent
        self.labels_set=set(range(len(self.label_name_list)))
        self.imglist = self.get_image_label_list()
        self.labels = np.array([e[1] for e in self.imglist])
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        img_name = self.imglist[idx][0]
        img=io.imread(img_name)
        img = np.array(img)
        if img.ndim==3:
            if img.shape[2]==4:
                img=color.rgba2rgb(img)
        if img.shape[0] > self.size:
            img=img[(img.shape[0]-self.size)//2:(img.shape[0]-self.size)//2+self.size, (img.shape[1]-self.size)//2:(img.shape[1]-self.size)//2+self.size,:]
        label = self.imglist[idx][1]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label # numpy image or torch depends on transform
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        labels=[]
        for dirpath, subdirs, files in os.walk(self.root_dir):
            for x in files:
                if x.endswith(('.png','.jpg')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    labels.append(x.split('_')[0])
        return names, wholePathes, labels
    
    def get_image_label_list(self):  
        f_name, f_path, l_name = self.walk_root_dir()
        image_label_list =  [[f_path[i], self.label_name_list.index(l_name[i])] for i,f in enumerate(f_name)]
        return image_label_list
    
    def statistic(self):
        return {self.label_name_list[key] : len(self.label_to_indices[key]) for key in self.label_to_indices.keys()}
    
class RetinaPIL(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """
    
    FILE = '/mnt/DATA_OTHER/retina/RFMiD_Training_Labels.csv'
    

    def __init__(self, root_dir, size=700, data_aug = True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.size = size
        self.data_aug = data_aug
        self.transform = transform
        
        self.imglist = self.get_image_label_list()
        self.labels = np.array([e[1] for e in self.imglist])
        
        self.label_name_list =list(sorted(set(map(str, list(self.labels))))) # sorted to keep consistent
        self.labels_set=set(range(len(self.label_name_list)))
        
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        img_name = self.imglist[idx][0]
        img = Image.open(img_name)
        img = img.resize((700, 460))
        if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img = t(img)
        img = np.array(img)
        img = img.astype('float32')
        if img.ndim==3:
            if img.shape[2]==4:
                img=color.rgba2rgb(img)
        if img.shape[0] > self.size:
            img=img[(img.shape[0]-self.size)//2:(img.shape[0]-self.size)//2+self.size, (img.shape[1]-self.size)//2:(img.shape[1]-self.size)//2+self.size,:]
        label = self.imglist[idx][1] 
#         slide = self.imglist[idx][2]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name  # numpy image or torch depends on transform
    
    def get_ground_truth(self, fid):
        interestingrow = []
        with open(RetinaPIL.FILE, newline='') as f:
            reader = csv.reader(f)
            interestingrow = [row for idx, row in enumerate(reader) if idx == fid]
        return list(map(int, interestingrow[0][1:])) 
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        for dirpath, subdirs, files in os.walk(self.root_dir):
            for x in files:
                if x.endswith(('.png','.jpeg')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))

        return names, wholePathes
    
    def get_image_label_list(self):  
        f_name, f_path = self.walk_root_dir()
        image_label_list =  [[f_path[i], self.get_ground_truth(int(f_name[i].split('.')[0]))[0], self.get_ground_truth(int(f_name[i].split('.')[0]))[1:]] for i,f in enumerate(f_name)]
        return image_label_list
    
    def statistic(self):
        return {self.label_name_list[key] : len(self.label_to_indices[key]) for key in self.label_to_indices.keys()}

class BreakHisPIL(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """

    def __init__(self, root_dir, size=700, data_aug = True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.size = size
        self.data_aug = data_aug
        self.transform = transform
        
        self.label_name_list =list(sorted(set(self.walk_root_dir()[2]))) # sorted to keep consistent
        self.labels_set=set(range(len(self.label_name_list)))
        
        self.slide_name_list =list(sorted(set(self.walk_root_dir()[3]))) # sorted to keep consistent
        self.slides_set=set(range(len(self.slide_name_list)))
        
        self.imglist = self.get_image_label_list()
        self.labels = np.array([e[1] for e in self.imglist])
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        img_name = self.imglist[idx][0]
        img = Image.open(img_name)
        img = img.resize((700, 460))
        if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img = t(img)
        img = np.array(img)
        img = img.astype('float32')
        if img.ndim==3:
            if img.shape[2]==4:
                img=color.rgba2rgb(img)
        if img.shape[0] > self.size:
            img=img[(img.shape[0]-self.size)//2:(img.shape[0]-self.size)//2+self.size, (img.shape[1]-self.size)//2:(img.shape[1]-self.size)//2+self.size,:]
        label = self.imglist[idx][1] 
        slide = self.imglist[idx][2]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label, slide, img_name  # numpy image or torch depends on transform
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        labels=[]
        slides=[]
        for dirpath, subdirs, files in os.walk(self.root_dir):
            for x in files:
                if x.endswith(('.png','.jpg')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    labels.append(x.split('_')[1]) # index 1 for labels
                    slides.append(x.split('_')[2].split('-')[2])
        return names, wholePathes, labels, slides
    
    def get_image_label_list(self):  
        f_name, f_path, l_name, s_name = self.walk_root_dir()
        image_label_list =  [[f_path[i], self.label_name_list.index(l_name[i]), self.slide_name_list.index(s_name[i])] for i,f in enumerate(f_name)]
        return image_label_list
    
    def statistic(self):
        return {self.label_name_list[key] : len(self.label_to_indices[key]) for key in self.label_to_indices.keys()}
    
class BladderPIL(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """

    def __init__(self, root_dir, size=299, data_aug = True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.size = size
        self.data_aug = data_aug
        self.transform = transform
        
        self.label_name_list =list(sorted(set(self.walk_root_dir()[2]))) # sorted to keep consistent
        self.labels_set=set(range(len(self.label_name_list)))
        
#         self.slide_name_list =list(sorted(set(self.walk_root_dir()[3]))) # sorted to keep consistent
#         self.slides_set=set(range(len(self.slide_name_list)))
        
        self.imglist = self.get_image_label_list()
        self.labels = np.array([e[1] for e in self.imglist])
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        img_name = self.imglist[idx][0]
        img = Image.open(img_name)
#         img = img.resize((700, 460))
        if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img = t(img)
        img = np.array(img)
        img = img.astype('float32')
        if img.ndim==3:
            if img.shape[2]==4:
                img=color.rgba2rgb(img)
        if img.shape[0] > self.size:
            img=img[(img.shape[0]-self.size)//2:(img.shape[0]-self.size)//2+self.size, (img.shape[1]-self.size)//2:(img.shape[1]-self.size)//2+self.size,:]
        label = self.imglist[idx][1] 
#         slide = self.imglist[idx][2]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name  # numpy image or torch depends on transform
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        labels=[]
        slides=[]
        for dirpath, subdirs, files in os.walk(self.root_dir):
            for x in files:
                if x.endswith(('.png','.jpeg')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    labels.append(x.split('_')[0]) # index 0 for labels
#                     slides.append(x.split('_')[2].split('-')[2])
        return names, wholePathes, labels
    
    def get_image_label_list(self):  
        f_name, f_path, l_name = self.walk_root_dir()
        image_label_list =  [[f_path[i], self.label_name_list.index(l_name[i])] for i,f in enumerate(f_name)]
        return image_label_list
    
    def statistic(self):
        return {self.label_name_list[key] : len(self.label_to_indices[key]) for key in self.label_to_indices.keys()}


class BaseHistoFromSampler(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """

    def __init__(self, imgs, imgs_w, size = 224, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.imgs = imgs
        self.imgs_w = imgs_w
        self.size = size
        self.transform = transform
        self.label_name_list =list(sorted(set([f.split('_')[0] for f in self.imgs]))) # sorted to keep consistent
        self.labels_set=set(range(len(self.label_name_list)))
        self.imglist = self.get_image_label_list()
        self.labels = np.array([e[1] for e in self.imglist])
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

    def __len__(self):
#         return len(self.get_img_list_1d()[0])
        return len(self.imgs)
    
    def get_img_list_1d(self):
        imglist1d=[]
        img_w_list1d=[]
        for i in range(self.imgs.shape[0]):
            for j in range(self.imgs.shape[1]):
                for f1 in self.imgs[i,j]:
                    imglist1d.append(f1)
                for f2 in self.imgs_w[i,j]:
                    img_w_list1d.append(f2)
                
        return imglist1d, img_w_list1d
        
            

    def __getitem__(self, idx):
        img_name = self.imglist[idx][0]
        img=io.imread(img_name)
        img = np.array(img)
        if img.ndim==3:
            if img.shape[2]==4:
                img=color.rgba2rgb(img).astype(np.float32)
        
        if (img.shape[0] < self.size):
            if (self.size - img.shape[0])%2 ==0:
                pad_h = (self.size - img.shape[0])//2
                img=np.pad(img, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
            else:
                pad_h = (self.size - img.shape[0])//2
                img=np.pad(img, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
        if (img.shape[1] < self.size):
            if (self.size - img.shape[1])%2 ==0:
                pad_h = (self.size - img.shape[1])//2
                img=np.pad(img, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
            else:
                pad_h = (self.size - img.shape[1])//2
                img=np.pad(img, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')
                
        if (img.shape[0] > self.size):
            img=img[(img.shape[0]-self.size)//2:(img.shape[0]-self.size)//2+self.size, :,:]
        if (img.shape[1] > self.size):
            img=img[:, (img.shape[1]-self.size)//2:(img.shape[1]-self.size)//2+self.size,:]
            
        label = self.imglist[idx][1]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label # numpy or torch depends on transform
    
    
    
    def get_image_label_list(self):
        image_label_list =  [[self.imgs_w[i], self.label_name_list.index(f.split('_')[0])] for i,f in enumerate(self.imgs)]
#         image_label_list =  [[self.get_img_list_1d()[1][i], self.label_name_list.index(f.split('_')[0])] for i,f in enumerate(self.get_img_list_1d()[0])]
        return image_label_list
    
    def statistic(self):
        return {self.label_name_list[key] : len(self.label_to_indices[key]) for key in self.label_to_indices.keys()}
    
class SiameseHisto(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, baseHisto, train, size = 224):
        self.histoDataset=baseHisto
        self.train = train
        self.size = size
        self.transform = self.histoDataset.transform
        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
            

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2) # random dicide the target
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            if img1.shape[0] > self.size:
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_name = self.train_data[siamese_index]
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            if img2.shape[0] > self.size:
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
        else:
            img1_name = self.test_data[self.test_pairs[index][0]]
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            if img1.shape[0] > self.size:
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            img2_name = self.test_data[self.test_pairs[index][1]]
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            if img2.shape[0] > self.size:
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
            target = self.test_pairs[index][2]
            
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.histoDataset)
    

class SiameseClassificationHisto(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, baseHisto, train, size = 224):
        self.histoDataset=baseHisto
        self.train = train
        self.size = size
        self.transform = self.histoDataset.transform
        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
            

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            if img1.shape[0] > self.size:
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_name, label2 = self.train_data[siamese_index], self.train_labels[siamese_index].item()
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            if img2.shape[0] > self.size:
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
        else:
            img1_name, label1 = self.test_data[self.test_pairs[index][0]], self.test_labels[self.test_pairs[index][0]]
            
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            if img1.shape[0] > self.size:
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            img2_name, label2 = self.test_data[self.test_pairs[index][1]], self.test_labels[self.test_pairs[index][1]]
            
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            if img2.shape[0] > self.size:
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
            target = self.test_pairs[index][2]
            
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), (target, label1)

    def __len__(self):
        return len(self.histoDataset)
    
class TripletHisto(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative sample
    Test: Creates fixed triplets for testing
    """

    def __init__(self, baseHisto, train, size=224):
        self.histoDataset = baseHisto
        self.train = train
        self.size = size
        self.transform = self.histoDataset.transform

        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            # generate fixed triplets for testing
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            if img1.shape[0] > self.size:
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_name = self.train_data[positive_index]
            
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            if img2.shape[0] > self.size:
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
            img3_name = self.train_data[negative_index]
            
            img3=io.imread(img3_name)
            img3 = np.array(img3)
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)
            if img3.shape[0] > self.size:
                img3=img3[(img3.shape[0]-self.size)//2:(img3.shape[0]-self.size)//2+self.size, (img3.shape[1]-self.size)//2:(img3.shape[1]-self.size)//2+self.size,:]
        else:
            img1_name = self.test_data[self.test_triplets[index][0]]
            
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            if img1.shape[0] > self.size:
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            img2_name = self.test_data[self.test_triplets[index][1]]
            
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            if img2.shape[0] > self.size:
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
            img3_name = self.test_data[self.test_triplets[index][2]]
            
            img3=io.imread(img3_name)
            img3 = np.array(img3)
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)
            if img3.shape[0] > self.size:
                img3=img3[(img3.shape[0]-self.size)//2:(img3.shape[0]-self.size)//2+self.size, (img3.shape[1]-self.size)//2:(img3.shape[1]-self.size)//2+self.size,:]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.histoDataset)
    
class TripletClassificationHisto(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, baseHisto, train, size=224):
        self.histoDataset = baseHisto
        self.train = train
        self.size = size
        self.transform = self.histoDataset.transform

        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            # generate fixed triplets for testing
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
                    
            if (img1.shape[0] < self.size):
                if (self.size - img1.shape[0])%2 ==0:
                    pad_h = (self.size - img1.shape[0])//2
                    img1=np.pad(img1, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img1.shape[0])//2
                    img1=np.pad(img1, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
            if (img1.shape[1] < self.size):
                if (self.size - img1.shape[1])%2 ==0:
                    pad_h = (self.size - img1.shape[1])//2
                    img1=np.pad(img1, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img1.shape[1])//2
                    img1=np.pad(img1, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')

            if (img1.shape[0] > self.size):
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, :,:]
            if (img1.shape[1] > self.size):
                img1=img1[:, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_name = self.train_data[positive_index]
            
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
                    
            if (img2.shape[0] < self.size):
                if (self.size - img2.shape[0])%2 ==0:
                    pad_h = (self.size - img2.shape[0])//2
                    img2=np.pad(img2, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img2.shape[0])//2
                    img2=np.pad(img2, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
            if (img2.shape[1] < self.size):
                if (self.size - img2.shape[1])%2 ==0:
                    pad_h = (self.size - img2.shape[1])//2
                    img2=np.pad(img2, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img2.shape[1])//2
                    img2=np.pad(img2, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')

            if (img2.shape[0] > self.size):
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, :,:]
            if (img2.shape[1] > self.size):
                img2=img2[:, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
            
            img3_name = self.train_data[negative_index]
            
            img3=io.imread(img3_name)
            img3 = np.array(img3)
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)
                    
            if (img3.shape[0] < self.size):
                if (self.size - img3.shape[0])%2 ==0:
                    pad_h = (self.size - img3.shape[0])//2
                    img3=np.pad(img3, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img3.shape[0])//2
                    img3=np.pad(img3, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
            if (img3.shape[1] < self.size):
                if (self.size - img3.shape[1])%2 ==0:
                    pad_h = (self.size - img3.shape[1])//2
                    img3=np.pad(img3, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img3.shape[1])//2
                    img3=np.pad(img3, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')

            if (img3.shape[0] > self.size):
                img3=img3[(img3.shape[0]-self.size)//2:(img3.shape[0]-self.size)//2+self.size, :,:]
            if (img3.shape[1] > self.size):
                img3=img3[:, (img3.shape[1]-self.size)//2:(img3.shape[1]-self.size)//2+self.size,:]
            
        else:
            img1_name = self.test_data[self.test_triplets[index][0]]
            label1 = self.test_labels[self.test_triplets[index][0]].item()
            
            img1=io.imread(img1_name)
            img1 = np.array(img1)
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            if (img1.shape[0] < self.size):
                if (self.size - img1.shape[0])%2 ==0:
                    pad_h = (self.size - img1.shape[0])//2
                    img1=np.pad(img1, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img1.shape[0])//2
                    img1=np.pad(img1, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
            if (img1.shape[1] < self.size):
                if (self.size - img1.shape[1])%2 ==0:
                    pad_h = (self.size - img1.shape[1])//2
                    img1=np.pad(img1, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img1.shape[1])//2
                    img1=np.pad(img1, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')

            if (img1.shape[0] > self.size):
                img1=img1[(img1.shape[0]-self.size)//2:(img1.shape[0]-self.size)//2+self.size, :,:]
            if (img1.shape[1] > self.size):
                img1=img1[:, (img1.shape[1]-self.size)//2:(img1.shape[1]-self.size)//2+self.size,:]
            
            img2_name = self.test_data[self.test_triplets[index][1]]
            
            img2=io.imread(img2_name)
            img2 = np.array(img2)
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
                    
            if (img2.shape[0] < self.size):
                if (self.size - img2.shape[0])%2 ==0:
                    pad_h = (self.size - img2.shape[0])//2
                    img2=np.pad(img2, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img2.shape[0])//2
                    img2=np.pad(img2, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
            if (img2.shape[1] < self.size):
                if (self.size - img2.shape[1])%2 ==0:
                    pad_h = (self.size - img2.shape[1])//2
                    img2=np.pad(img2, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img2.shape[1])//2
                    img2=np.pad(img2, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')

            if (img2.shape[0] > self.size):
                img2=img2[(img2.shape[0]-self.size)//2:(img2.shape[0]-self.size)//2+self.size, :,:]
            if (img2.shape[1] > self.size):
                img2=img2[:, (img2.shape[1]-self.size)//2:(img2.shape[1]-self.size)//2+self.size,:]
            
            img3_name = self.test_data[self.test_triplets[index][2]]
            
            img3=io.imread(img3_name)
            img3 = np.array(img3)
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)
                    
            if (img3.shape[0] < self.size):
                if (self.size - img3.shape[0])%2 ==0:
                    pad_h = (self.size - img3.shape[0])//2
                    img3=np.pad(img3, ((pad_h,pad_h), (0,0), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img3.shape[0])//2
                    img3=np.pad(img3, ((pad_h,pad_h+1), (0,0), (0,0)), 'constant')
            if (img3.shape[1] < self.size):
                if (self.size - img3.shape[1])%2 ==0:
                    pad_h = (self.size - img3.shape[1])//2
                    img3=np.pad(img3, ((0,0), (pad_h,pad_h), (0,0)), 'constant')
                else:
                    pad_h = (self.size - img3.shape[1])//2
                    img3=np.pad(img3, ((0,0), (pad_h,pad_h+1), (0,0)), 'constant')

            if (img3.shape[0] > self.size):
                img3=img3[(img3.shape[0]-self.size)//2:(img3.shape[0]-self.size)//2+self.size, :,:]
            if (img3.shape[1] > self.size):
                img3=img3[:, (img3.shape[1]-self.size)//2:(img3.shape[1]-self.size)//2+self.size,:]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), label1

    def __len__(self):
        return len(self.histoDataset)
    
class SiameseClsBreakHisPIL(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, baseHisto, train):
        self.histoDataset=baseHisto
        self.train = train
        self.transform = self.histoDataset.transform
        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
            

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            img1 = Image.open(img1_name)
            img1 = img1.resize((700, 460))
#             if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img1 = t(img1)
            
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_name, label2 = self.train_data[siamese_index], self.train_labels[siamese_index].item()
            img2 = Image.open(img2_name)
            img2 = img2.resize((700, 460))
#             if self.data_aug:
            img2 = t(img2)
            
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
        else:
            img1_name, label1 = self.test_data[self.test_pairs[index][0]], self.test_labels[self.test_pairs[index][0]]
            img1 = Image.open(img1_name)
            img1 = img1.resize((700, 460))
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
           
            img2_name, label2 = self.test_data[self.test_pairs[index][1]], self.test_labels[self.test_pairs[index][1]]
            img2 = Image.open(img2_name)
            img2 = img2.resize((700, 460))
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
            target = self.test_pairs[index][2]
            
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), (target, label1)

    def __len__(self):
        return len(self.histoDataset)
    
class SiameseClsPIL(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, baseHisto, train):
        self.histoDataset=baseHisto
        self.train = train
        self.transform = self.histoDataset.transform
        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
            

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            img1 = Image.open(img1_name)
#             img1 = img1.resize((700, 460))
#             if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img1 = t(img1)
            
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_name, label2 = self.train_data[siamese_index], self.train_labels[siamese_index].item()
            img2 = Image.open(img2_name)
#             img2 = img2.resize((700, 460))
#             if self.data_aug:
            img2 = t(img2)
            
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
        else:
            img1_name, label1 = self.test_data[self.test_pairs[index][0]], self.test_labels[self.test_pairs[index][0]]
            img1 = Image.open(img1_name)
#             img1 = img1.resize((700, 460))
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
           
            img2_name, label2 = self.test_data[self.test_pairs[index][1]], self.test_labels[self.test_pairs[index][1]]
            img2 = Image.open(img2_name)
#             img2 = img2.resize((700, 460))
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
            target = self.test_pairs[index][2]
            
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), (target, label1)

    def __len__(self):
        return len(self.histoDataset)
    
class TripletClsBreakHisPIL(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, baseHisto, train):
        self.histoDataset = baseHisto
        self.train = train
        self.transform = self.histoDataset.transform

        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            # generate fixed triplets for testing
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            
            img1 = Image.open(img1_name)
            img1 = img1.resize((700, 460))
#             if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img1 = t(img1)
            
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_name = self.train_data[positive_index]
            
            img2 = Image.open(img2_name)
            img2 = img2.resize((700, 460))
            img2 = t(img2)
            
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
            img3_name = self.train_data[negative_index]
            
            img3 = Image.open(img3_name)
            img3 = img3.resize((700, 460))
            img3 = t(img3)
            
            img3 = np.array(img3)
            img3 = img3.astype('float32')
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)
            
        else:
            img1_name = self.test_data[self.test_triplets[index][0]]
            label1 = self.test_labels[self.test_triplets[index][0]].item()
            
            img1 = Image.open(img1_name)
            img1 = img1.resize((700, 460))
            
            
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            img2_name = self.test_data[self.test_triplets[index][1]]
            
            img2 = Image.open(img2_name)
            img2 = img2.resize((700, 460))
           
            
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
            img3_name = self.test_data[self.test_triplets[index][2]]
            
            img3 = Image.open(img3_name)
            img3 = img3.resize((700, 460))
            
            
            img3 = np.array(img3)
            img3 = img3.astype('float32')
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), label1

    def __len__(self):
        return len(self.histoDataset)
    
class TripletClsPIL(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, baseHisto, train):
        self.histoDataset = baseHisto
        self.train = train
        self.transform = self.histoDataset.transform

        if self.train:
            self.train_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.train_data = np.array([e[0] for e in self.histoDataset.imglist])
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = np.array([e[1] for e in self.histoDataset.imglist])
            self.test_data = np.array([e[0] for e in self.histoDataset.imglist])
            # generate fixed triplets for testing
            self.labels_set = self.histoDataset.labels_set
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1_name, label1 = self.train_data[index], self.train_labels[index].item()
            
            img1 = Image.open(img1_name)
#             img1 = img1.resize((700, 460))
#             if self.data_aug:
            hf = RandomHorizontalFlip()
            vf = RandomVerticalFlip()
            cj = ColorJitter(0.1,0.1,0.1,0.1)
            t = Compose([hf, vf, cj])
            img1 = t(img1)
            
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_name = self.train_data[positive_index]
            
            img2 = Image.open(img2_name)
#             img2 = img2.resize((700, 460))
            img2 = t(img2)
            
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
            img3_name = self.train_data[negative_index]
            
            img3 = Image.open(img3_name)
#             img3 = img3.resize((700, 460))
            img3 = t(img3)
            
            img3 = np.array(img3)
            img3 = img3.astype('float32')
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)
            
        else:
            img1_name = self.test_data[self.test_triplets[index][0]]
            label1 = self.test_labels[self.test_triplets[index][0]].item()
            
            img1 = Image.open(img1_name)
#             img1 = img1.resize((700, 460))
            
            
            img1 = np.array(img1)
            img1 = img1.astype('float32')
            if img1.ndim==3:
                if img1.shape[2]==4:
                    img1=color.rgba2rgb(img1).astype(np.float32)
            
            img2_name = self.test_data[self.test_triplets[index][1]]
            
            img2 = Image.open(img2_name)
#             img2 = img2.resize((700, 460))
           
            
            img2 = np.array(img2)
            img2 = img2.astype('float32')
            if img2.ndim==3:
                if img2.shape[2]==4:
                    img2=color.rgba2rgb(img2).astype(np.float32)
            
            img3_name = self.test_data[self.test_triplets[index][2]]
            
            img3 = Image.open(img3_name)
#             img3 = img3.resize((700, 460))
            
            
            img3 = np.array(img3)
            img3 = img3.astype('float32')
            if img3.ndim==3:
                if img3.shape[2]==4:
                    img3=color.rgba2rgb(img3).astype(np.float32)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), label1

    def __len__(self):
        return len(self.histoDataset)