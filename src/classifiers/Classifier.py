from abc import ABC, abstractmethod 
from neurorobotics_dl.prototype_learning import PrototypicalModel
from neurorobotics_dl.utils import get_class
import os
import numpy as np
from joblib import load
import torch

class Classifier(ABC):
    @abstractmethod
    def __call__(self,x):
        pass

class ProtypicalClassifier(Classifier):
    def __init__(self,cfg):
        sbj_path = cfg['path']
        model_path = os.path.join(sbj_path,"model.pt")
        clf_path = os.path.join(sbj_path,"qda.joblib")
        mean_std_path = os.path.join(sbj_path,"mean_std.npz")
        self.net = get_class(cfg['model']['classname'])(**cfg['model']['options']).to('cpu')
        self.net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.clf = load(clf_path) 
        cc = np.load(mean_std_path)
        self.mu, self.sigma = cc['mu'], cc['sigma']

    def __call__(self,x):
        x = (x-self.mu)/self.sigma
        emb = self.net.compute_embeddings(x)
        pred = [0 for _ in range(len(self.clf.classes_))]
        pred[int(self.clf.predict(emb))] = 1
        probs = self.clf.predict_proba(emb)[0].tolist()
        return pred, probs
    
class NeuralClassifier(Classifier):
    def __init__(self,cfg):
        sbj_path = cfg['path']
        model_path = os.path.join(sbj_path,"model.pt")
        mean_std_path = os.path.join(sbj_path,"mean_std.npz")
        self.net = get_class(cfg['classname'])(**cfg['options']).to('cpu')
        self.classes = cfg['classes']
        self.num_classes = len(self.classes)
        print(self.num_classes)

        self.net.eval()
        self.net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))['model'])
        cc = np.load(mean_std_path)
        self.mu, self.sigma = cc['mu'], cc['sigma']

    def __call__(self,x):
        x = (x-self.mu)/self.sigma
        with torch.no_grad():
            probs = self.net(x).cpu()[:,:self.num_classes]
        probs = torch.softmax(probs,dim=1).squeeze().tolist()
        pred = [0] * self.num_classes
        pred[int(np.argmax(probs))] = 1
        return pred, probs
    
    def get_classes(self):
        return self.classes


class NeuralClassifier(Classifier):
    def __init__(self,net,mu,sigma):
        self.net = net
        self.mu = mu
        self.sigma = sigma

    def from_config(cfg_path):
        # LE TENGO SOLO PER PARANOIA CHE QUALCOSA NON VADA MA DOVREI POTERLE BUTTARE
        # sbj_path = cfg['path']
        # model_path = os.path.join(sbj_path,"model.pt")
        # mean_std_path = os.path.join(sbj_path,"mean_std.npz")
        # self.net = get_class(cfg['classname'])(**cfg['options']).to('cpu')
        # self.classes = cfg['classes']
        # self.num_classes = len(self.classes)
        # print(self.num_classes)

        # self.net.eval()
        # self.net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))['model'])
        # cc = np.load(mean_std_path)
        # self.mu, self.sigma = cc['mu'], cc['sigma']

        cfg = torch.load(cfg_path,map_location=torch.device('cpu'))

        net = get_class(cfg['config']['classname'])(**cfg['config']['options']).to('cpu')
        net.load_state_dict(cfg['model'])
        net.eval()
        mu,sigma = cfg['mu'],cfg['sigma']
        return NeuralClassifier(net,mu,sigma)

    def __call__(self,x):
        x = (x-self.mu)/self.sigma
        with torch.no_grad():
            probs = self.net(x).cpu()
        probs = torch.softmax(probs,dim=1).squeeze().tolist()
        pred = [0] * len(probs)
        pred[int(np.argmax(probs))] = 1
        return pred, probs

