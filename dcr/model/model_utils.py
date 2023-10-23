import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F
import numpy as np


def mixup_data(x, att):

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index]
    
    return mixed_x, index


class CosineClassifier(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim,out_dim))
        
    def l2_normalize(self,x):
        x = x/torch.sum(x**2, dim=-1, keepdim=True).sqrt()
        return x

    def forward(self, x):
        t = x.shape[1]
        assert len(x.shape) == 3 and t == self.weight.shape[1]
        x = self.l2_normalize(x)
        w = self.l2_normalize(self.weight)
        return x @ w 


class ActionClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], dropout = 0., loss_smooth = 0, loss_weight = None):
        super().__init__()
        self.layers = []     
        for i,dim in enumerate(hiddens):
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_dim,dim))
            self.layers.append(nn.ReLU())
            in_dim = dim
        if dropout > 0:
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(in_dim,out_dim))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self,x,y = None):
        if y is None:
            return self.layers(x)
        else:
            x = x.reshape(-1,x.shape[-1])
            y = y.flatten()
            mask = (y>=0)
            x,y = x[mask], y[mask]
            y_pred = self.layers(x)
            
            return y_pred, y


class ActionClassifierV2(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], dropout = 0., loss_smooth = 0, loss_weight = None):
        super().__init__()
        self.layers = []
        for i,dim in enumerate(hiddens):
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_dim,dim))
            self.layers.append(nn.ReLU())
            in_dim = dim
        if dropout > 0:
            #self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*self.layers)

        self.classifier = nn.Linear(in_dim,out_dim)
        self.uncertainty_layer = nn.Linear(in_dim,out_dim)

    def forward(self, x, y=None):
        
        if y is None:
            predicted_features = self.layers(x)
                        
            ## log uncertainty prediction
            predicted_uncertainties = self.uncertainty_layer(predicted_features)
            predicted_uncertainties = predicted_uncertainties.exp() # [bsz, dim]
            predicted_uncertainties = predicted_uncertainties.mean(dim=1, keepdim=True)

            ## classification
            y_pred = self.classifier(predicted_features)
            y_pred = y_pred / F.sigmoid(predicted_uncertainties)
            
            return y_pred
        else:
            # import ipdb; ipdb.set_trace()
            x = x.reshape(-1, x.shape[-1])
            y = y.flatten()
            mask = (y>=0)
            x, y = x[mask], y[mask]

            predicted_features = self.layers(x)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            ## log uncertainty prediction
            predicted_uncertainties = self.uncertainty_layer(predicted_features)
            predicted_uncertainties = predicted_uncertainties.exp() # [bsz, dim]
            
            ## classification
            y_pred = self.classifier(x)
            
            return y_pred, y, predicted_uncertainties
            

class ActionClassifierWithLoss_Vis(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], dropout = 0., loss_smooth = 0, loss_weight = None):
        super().__init__()
        self.layers = []
        for i,dim in enumerate(hiddens):
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_dim,dim))
            self.layers.append(nn.ReLU())
            in_dim = dim
        if dropout > 0:
            #self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*self.layers)

        self.classifier = nn.Linear(in_dim, out_dim)
        self.uncertainty_layer = nn.Linear(in_dim, out_dim)
        
        # self.loss_func = CrossEntropy(loss_smooth,loss_weight)

    def forward(self, x, y=None):
        
        if y is None:
            predicted_features = self.layers(x)
                        
            ## log uncertainty prediction
            predicted_uncertainties = self.uncertainty_layer(predicted_features)
            predicted_uncertainties = predicted_uncertainties.exp() # [bsz, dim]
            predicted_uncertainties = predicted_uncertainties.mean(dim=1, keepdim=True)

            ## classification
            y_pred = self.classifier(predicted_features)
            y_pred = y_pred / predicted_uncertainties
            
            return y_pred, predicted_uncertainties
            
        else:
            # import ipdb; ipdb.set_trace()
            x = x.reshape(-1, x.shape[-1])
            y = y.flatten()
            mask = (y>=0)
            x, y = x[mask], y[mask]

            predicted_features = self.layers(x)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            ## log uncertainty prediction
            predicted_uncertainties = self.uncertainty_layer(predicted_features)
            predicted_uncertainties = predicted_uncertainties.exp() # [bsz, dim]
            
            ## mixup
            mixed_features, mixup_index = mixup_data(predicted_features, predicted_uncertainties.mean(dim=1, keepdim=True)) # [bsz, dim]
            
            ## classification
            y_pred = self.classifier(mixed_features)
            
            return y_pred, y, predicted_uncertainties, mixup_index
        

class ActionClassifierWithLoss(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], dropout = 0., loss_smooth = 0, loss_weight = None, mixup=True):
        super().__init__()
        self.layers = []
        for i,dim in enumerate(hiddens):
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_dim,dim))
            self.layers.append(nn.ReLU())
            in_dim = dim
        if dropout > 0:
            #self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*self.layers)

        self.classifier = nn.Linear(in_dim, out_dim)
        self.uncertainty_layer = nn.Linear(in_dim, out_dim)
        self.mixup = mixup
        # self.loss_func = CrossEntropy(loss_smooth,loss_weight)

    def forward(self, x, y=None):
        
        if y is None:
            predicted_features = self.layers(x)
                        
            ## log uncertainty prediction
            predicted_uncertainties = self.uncertainty_layer(predicted_features)
            predicted_uncertainties = predicted_uncertainties.exp() # [bsz, dim]
            predicted_uncertainties = predicted_uncertainties.mean(dim=1, keepdim=True)

            ## classification
            y_pred = self.classifier(predicted_features)
            y_pred = y_pred / predicted_uncertainties
            #y_pred = y_pred / F.sigmoid(predicted_uncertainties)
            
            return y_pred
            # return self.layers(x)
        else:
            # import ipdb; ipdb.set_trace()
            x = x.reshape(-1, x.shape[-1])
            y = y.flatten()
            mask = (y>=0)
            x, y = x[mask], y[mask]

            predicted_features = self.layers(x)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            ## log uncertainty prediction
            predicted_uncertainties = self.uncertainty_layer(predicted_features)
            predicted_uncertainties = predicted_uncertainties.exp() # [bsz, dim]
            
            if self.mixup:
                ## mixup
                mixed_features, mixup_index = mixup_data(predicted_features, predicted_uncertainties.mean(dim=1, keepdim=True)) # [bsz, dim]
                
                ## classification
                y_pred = self.classifier(mixed_features)
                
                return y_pred, y, predicted_uncertainties, mixup_index
            else: 
                ## classification
                y_pred = self.classifier(predicted_features)
                
                return y_pred, y, predicted_uncertainties

class ClassifierWithLoss(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], dropout = 0., loss_smooth = 0, loss_weight = None):
        super().__init__()
        self.layers = []     
        for i,dim in enumerate(hiddens):
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_dim,dim))
            self.layers.append(nn.ReLU())
            in_dim = dim
        if dropout > 0:
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(in_dim,out_dim))
        self.layers = nn.Sequential(*self.layers)
        self.loss_func = CrossEntropy(loss_smooth,loss_weight)

    def forward(self,x,y = None):
        if y is None:
            return self.layers(x)
        else:
            x = x.reshape(-1,x.shape[-1])
            y = y.flatten()
            mask = (y>=0)
            x,y = x[mask], y[mask]
            y_pred = self.layers(x)
            loss = self.loss_func(y_pred,y)
            return loss


class CrossEntropy(nn.Module):
    def __init__(self,smooth = 0.,weight = None):
        super().__init__()
        self.smooth = smooth
        self.weight = weight

        if self.weight is None:
            pass
        elif isinstance(self.weight, str):
            self.weight = nn.Parameter(
                torch.tensor(pkl.load(open(weight,'rb'))).float(), requires_grad=False)
        elif isinstance(self.weight, np.ndarray):
            self.weight = nn.Parameter(
                torch.tensor(self.weight).float(), requires_grad=False)
        elif isinstance(self.weight, torch.Tensor):
            self.weight = nn.Parameter(self.weight.float(), requires_grad=False)
        else:
            raise TypeError(weight)

    def forward(self,x,target):
        if target.shape!= x.shape:
            target = F.one_hot(target,num_classes = x.shape[-1])
        if self.weight is not None:
            target = target * self.weight
        if self.smooth > 0:
            num_cls = x.shape[-1]
            target = target * (1-self.smooth) + self.smooth/num_cls
        loss = - target * F.log_softmax(x, dim=-1)
        return loss.mean(0).sum()
