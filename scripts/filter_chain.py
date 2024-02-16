import numpy as np
import rospy
from rosneuro_msgs.msg import NeuroFrame
from scipy.io import loadmat
from scipy.signal import butter, lfilter, lfilter_zi

# import rosbag
# from std_msgs.msg import Float32MultiArray, MultiArrayDimension

ORDER_LP = 2
ORDER_HP = 2
CUTOFF_LP = 40
CUTOFF_HP = 2
SAMPLE_RATE = 512

class FilterChain:

    def __init__(self,cfg):
        
        self.cutoff_high = cfg['cutoff_high']
        self.cutoff_low = cfg['cutoff_low']
        self.fs = cfg['samplerate']
        self.order = cfg['order']

        high = (2*self.cutoff_high)/self.fs
        low = (2*self.cutoff_low)/self.fs
        self.b_bp, self.a_bp = butter(self.order, [low, high], btype='band')
        
        self.lap = loadmat(cfg['lap_path'])['lapmask']
        self.states = None

    def __call__(self,data):
        
        if self.states is None: # Set up node on first iteration
            self.nChans = data.shape[1]
            self.states = [lfilter_zi(self.b_bp,self.a_bp) for _ in range(self.nChans)]

        for ch in range(self.nChans):
            data[:,ch], self.states[ch], = lfilter(self.b_bp,self.a_bp,data[:,ch],zi = self.states[ch])
        data = np.matmul(data,self.lap)
        return data


        
def callback(data: NeuroFrame):
    global filter_chain,pub
    eeg = np.array(data.eeg.data).reshape(data.eeg.info.nsamples,data.eeg.info.nchannels)
    eeg_data = filter_chain(eeg)

    data.eeg.data = eeg_data.flatten().tolist()
    pub.publish(data)

    # global raw_bag,filtered_bag

    # new_msg = Float32MultiArray()
    # new_msg.layout.dim = [MultiArrayDimension()]
    # new_msg.data = data.eeg.data
    # raw_bag.write('raw',new_msg)

    # new_msg = Float32MultiArray()
    # new_msg.layout.dim = [MultiArrayDimension()]
    # new_msg.data = eeg_data.flatten().tolist()
    # filtered_bag.write('filtered',new_msg)
  
def main():
    global new_data, seq, pub
    global filter_chain
    new_data = False
    seq = 0
    
    # Init the node
    node_name = 'filterchain_node'
    rospy.init_node(node_name)
    hz = rospy.get_param('rate', 16) # data.sr / nsample
    
    # Init the Publisher
    rospy.Subscriber('neurodata', NeuroFrame, callback)
    pub = rospy.Publisher('neurodata_filtered', NeuroFrame, queue_size=1)
    # Setup the filter chain
    cfg = rospy.get_param(rospy.get_param(f'/{node_name}/configname'))
    filter_chain=FilterChain(cfg)

    # global raw_bag,filtered_bag
    # raw_bag = rosbag.Bag('/home/curtaz/Neurorobotics/rawdata_python.bag', 'w')
    # filtered_bag = rosbag.Bag('/home/curtaz/Neurorobotics/filtdata_python.bag', 'w')

    # Spin until shutdown
    rospy.spin()

    # raw_bag.close()
    # filtered_bag.close()
        
if __name__ == '__main__':
  main()