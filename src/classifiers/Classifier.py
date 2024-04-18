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
        self.net.load_state_dict(torch.load(model_path))
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

        self.net.eval()
        self.net.load_state_dict(torch.load(model_path)['model'])
        cc = np.load(mean_std_path)
        self.mu, self.sigma = cc['mu'], cc['sigma']

    def __call__(self,x):
        x = (x-self.mu)/self.sigma
        with torch.no_grad():
            probs = self.net(x).cpu()[:,:2]
        probs = torch.softmax(probs,dim=1).squeeze().tolist()
        pred = [0,0]
        pred[int(np.argmax(probs))] = 1
        return pred, probs

