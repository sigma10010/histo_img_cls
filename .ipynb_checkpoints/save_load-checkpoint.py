import torch
import torch.backends.cudnn as cudnn
import os
def save_model(model, model_name, epoch):
    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    filename='./checkpoint/ckpt_%s_%05d.t13'%(model_name, epoch)
    state=model.state_dict()
    torch.save(state, filename)

def load_model(model, model_name, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    filename='./checkpoint/ckpt_%s_%05d.t13'%(model_name, epoch)
    model = model.to(device)
    if device == 'cuda':
#         model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    state = torch.load(filename)
    model.load_state_dict(state) 
    return model

def load_model_ckpt(model, ckpt):
    """
    ckpt: relative path
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    filename='./checkpoint/' + ckpt
    model = model.to(device)
    if device == 'cuda':
#         model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    state = torch.load(filename)
    model.load_state_dict(state) 
    return model

def load_model_ckpt_w(model, ckpt):
    """
    ckpt: whole path
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    filename= ckpt
    model = model.to(device)
    if device == 'cuda':
#         model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    state = torch.load(filename)
    model.load_state_dict(state) 
    return model