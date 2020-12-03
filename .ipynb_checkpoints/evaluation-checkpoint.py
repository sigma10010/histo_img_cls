from sklearn.metrics import confusion_matrix
import numpy as np

def get_scores(model, dataloader, cuda):
    scores = []
    for batch_idx, (data, target, _) in enumerate(dataloader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        outputs = model.get_classification(*data)
        scores.append(outputs)
    
    return torch.cat(scores,0).detech().cpu().numpy()

def conf_matrix_breakHis(model, dataloader, labels_set, cuda):
    c=np.zeros((len(labels_set),len(labels_set)),dtype=int)
    for batch_idx, (data, target, _, _) in enumerate(dataloader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        outputs = model.get_classification(*data)
        pred=outputs.max(1)[1]
        c+=confusion_matrix(target.cpu(),pred.cpu(),labels=labels_set)
    return c

def conf_matrix(model, dataloader, labels_set, cuda):
    c=np.zeros((len(labels_set),len(labels_set)),dtype=int)
    for batch_idx, (data, target) in enumerate(dataloader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        outputs = model.get_classification(*data)
        pred=outputs.max(1)[1]
        c+=confusion_matrix(target.cpu(),pred.cpu(),labels=labels_set)
    return c

def f1_score(confusion_matrix):
    f=np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        f[i]=2*confusion_matrix[i][i]/((confusion_matrix.sum(0)+confusion_matrix.sum(1))[i])
    return f

def accuracy(confusion_matrix):  
    t=0
    for i in range(confusion_matrix.shape[0]):
        t+=confusion_matrix[i][i]
    return t/confusion_matrix.sum()

def patient_score_matrix(model, dataloader, cuda):
    num_slide = len(dataloader.dataset.slide_name_list)
    ps_matrix = np.zeros((num_slide,2))
    miscls = []
    for batch_idx, (data, target, slide_id, im_name) in enumerate(dataloader):
        for i in slide_id.cpu().numpy():
            ps_matrix[i,1]+=1
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        outputs = model.get_classification(*data)
        pred=outputs.max(1)[1]
        mis_idx = (target!=pred).cpu().numpy()
        miscls.append([im_name[k] for k, value in enumerate(mis_idx) if value==True])
        
        _slide_id = slide_id[pred==target]
        for j in _slide_id.cpu().numpy():
            ps_matrix[j,0]+=1
            
    return ps_matrix, np.array(miscls)

    