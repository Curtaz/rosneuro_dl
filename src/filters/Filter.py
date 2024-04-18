from abc import ABC,abstractmethod
from scipy.io import loadmat
from scipy.signal import butter,lfilter,lfilter_zi
import numpy as np

class Filter(ABC):
    @abstractmethod
    def apply(self,buffer):
        pass

    def __call__(self,data):
        return self.apply(data)

class LaplacianFilter(Filter):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.lap = loadmat(cfg['lap_path'])['lapmask']
    def apply(self,data) -> None:
        return np.matmul(data,self.lap)

class ButterworthFilter(Filter):
    def __init__(self,cfg) -> None:
        super().__init__()
        fs = cfg['samplerate']
        order = cfg['order']
        cutoff = 2*np.array(cfg['cutoff'])/fs
        btype = cfg['btype']
        self.b, self.a = butter(order, cutoff, btype=btype)
        self.states = None

    def apply(self,data) -> None:
        if self.states is None: # Set up node on first iteration
            self.nChans = data.shape[1]
            self.states = [lfilter_zi(self.b,self.a) for _ in range(self.nChans)]
        for ch in range(self.nChans):
            data[:,ch], self.states[ch], = lfilter(self.b,self.a,data[:,ch],zi = self.states[ch])
        return data

class FilterChain(Filter):
    def __init__(self,cfg):
        super().__init__()
        butter = ButterworthFilter(cfg)
        lap = LaplacianFilter(cfg)
        self.filters = [lap,butter]
    
    def apply(self,data):
        for filter in self.filters:
            data = filter.apply(data)
        return data

    def insert_filter(self,filter,pos=-1):
        self.filters.insert(pos,filter)

    def remove_filter(self,pos):
        del(self.filters[pos])