# import data_processing as dp
from lib import utils
import numpy as np
import pandas as pd
from scipy import signal

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class Features(Dataset):
        def extract_features(self):
            ...
 
    """    
    def __init__(self,config):
        self.class_info = [{'id':0, 'name':'normal'},
                           {'id':1, 'name':'Shank Tremble'},
                           {'id':2, 'name':'Shuffling'},
                           {'id':3, 'name':'Festination'},
                           {'id':4, 'name':'Schrittabfolge'},
                           {'id':5, 'name':'Loss of Balance'},
                           {'id':6, 'name':'Akinesia'}]
        self.width = config.WINDOW_SIZE
        self.stride = config.STEP_SIZE
        self.channels = ['Time','Label1','Label2']+config.CHANNELS
        self.fn_lp = config.FN_LP
        self.fn_hp = config.FN_HP
        self.fn_ir = config.FN_IR
        self.scaler = config.SCALE
        self.same_label = config.SAME_LABEL
        self.drop_with_zscore = config.DROP_WITH_ZSCORE
        self.remove_freqs = config.REMOVE_FREQS
        self.Lambda = config.DETREND_LAMBDA
        self.test_files = config.TEST_FILES
        self.train_ratio = config.TRAIN_SET_RATIO

    def load_data(self,files,show_process=True):

        N = len(files)
        
        i = 0
        self.X = []
        self.Y = []
        self.X2 = []
        self.Y2 = []
        self.X3 = []
        self.Y3 = []
        self.F = []
        self.F2 = []
        self.F3 = []
        for file in files:
            i += 1
            if file.find('G04')==0:
                print('skip')
                continue
            # read data from csv files
            emg_data = pd.read_csv('./data/'+file)
            # drop out NA value and take data from selected channels
            emg_data = emg_data.loc[:,self.channels].dropna().reset_index(drop=True)
            
#             if (self.fn_hp != None) & (self.Lambda == None):
#                 wn=2*self.fn_hp/1000
#                 b, a = signal.butter(4, [wn], 'highpass')
#                 emg_data.iloc[:,3:] = signal.filtfilt(b,a,emg_data.iloc[:,3:],axis=0)
            
#             if self.fn_lp != None:
#                 wn=2*self.fn_lp/1000
#                 b, a = signal.butter(4, [wn], 'lowpass')
#                 emg_data.iloc[:,3:] = signal.filtfilt(b,a,emg_data.iloc[:,3:],axis=0)
                
#             if self.fn_ir:
#                 for c in range(len(self.channels[3:])):
#                     freqs, power=signal.periodogram(emg_data.iloc[:,c], 1e3)
#                     fmax = np.argmax(power)
#                     if (freqs[fmax] >49) & (freqs[fmax] <51):
#                         fs = 1000.0  # Sample frequency (Hz)
#                         f0 = 50  # Frequency to be removed from signal (Hz)
#                         Q = 200.0  # Quality factor
#                         # Design notch filter
#                         b1, a1 = signal.iirnotch(f0, Q, fs)
#                         emg_data.iloc[:,c] = signal.filtfilt(b1,a1,emg_data.iloc[:,c])
#                         print('fn_ir')

            x,y = utils.generate_window_slide_data_time_continue(emg_data, 
                                              width=self.width,
                                              stride=self.stride,
                                              scaler=self.scaler,
                                              same_label=self.same_label,
                                              drop_with_zscore=self.drop_with_zscore,
                                              remove_freqs=self.remove_freqs)
 
            shape = x.shape
            
            # skip empty dataset (can be caused by drop NA value)
            if shape[0]==0:
                print('skip')
                continue
            
            # use detrend methode (similar with highpass filter)
            if self.Lambda != None:
                x = utils.detrend(x,self.Lambda)
            # use highpass filter if lambda is not given
            elif self.fn_hp != None:
                wn=2*self.fn_hp/1000
                b, a = signal.butter(4, [wn], 'highpass')
                x = signal.filtfilt(b,a,x,axis=1)
            
            # use lowpass filter
            if self.fn_lp != None:
                wn=2*self.fn_lp/1000
                b, a = signal.butter(4, [wn], 'lowpass')
                x = signal.filtfilt(b,a,x,axis=1)
                
            if self.fn_ir:
                freqs, power=signal.periodogram(x, 1e3, axis=1)
                fmax = np.argmax(power,axis=1)
                freqs = np.broadcast_to(freqs[np.newaxis,:,np.newaxis], power.shape)
                fmax = np.take_along_axis(freqs,fmax[:,np.newaxis,:],axis=1)
                
                ind_temp = (fmax < 51) & (fmax > 49) & (np.max(power,axis=1)>0.1)[:,np.newaxis,:]
                f_ind = np.where(ind_temp)
                if np.any(np.any(ind_temp)):
                    print('fn_ir:',np.sum(ind_temp),' ',np.array(np.where(ind_temp))[[0,2]])
                fs = 1000.0  # Sample frequency (Hz)
                f0 = 50  # Frequency to be removed from signal (Hz)
                Q = 200.0  # Quality factor
                # Design notch filter
                b1, a1 = signal.iirnotch(f0, Q, fs)
                x[f_ind[0],:,f_ind[2]] = signal.filtfilt(b1,a1,x[f_ind[0],:,f_ind[2]],axis=1)

            ind1 = []
            ind2 = []
            ind3 = []

            for j in set(y):
                ind = np.where(y == j)[0].tolist()
                l_t = len(ind)
                if file in self.test_files:
                    ind3 += ind
                else:
                    ind1 += ind[:int(l_t*self.train_ratio)]
                    ind2 += ind[int(l_t*self.train_ratio):int(l_t*1.)]
                
            fi = [file]*len(ind1)
            fi2 = [file]*len(ind2)
            fi3 = [file]*len(ind3)

            self.X += x[ind1].tolist()
            self.Y += y[ind1].tolist()

            self.X2 += x[ind2].tolist()
            self.Y2 += y[ind2].tolist()

            self.X3 += x[ind3].tolist()
            self.Y3 += y[ind3].tolist()

            self.F += fi
            self.F2 += fi2
            self.F3 += fi3
            if show_process:
                print('%d/%d: '%(i,N)+file)

        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X2 = np.array(self.X2)
        self.Y2 = np.array(self.Y2)
        self.X3 = np.array(self.X3)
        self.Y3 = np.array(self.Y3)
    
    @property        
    def train_set(self):
        return self.X,self.Y,self.F
    
    @property
    def valid_set(self):
        return self.X2,self.Y2,self.F2
    
    @property
    def test_set(self):
        return self.X3,self.Y3,self.F3
    
        
class Features(Dataset):
    """
    Class for feature extraction.
    """
    
    def __init__(self,config):
        
        super(Features,self).__init__(config)
        
        self.threshold_WAMP = config.THRESHOLD_WAMP
        self.threshold_ZC = config.THRESHOLD_ZC
        self.threshold_SSC = config.THRESHOLD_SSC
        self.bins = config.BINS
        self.ranges = config.RANGES
        self.wavelet = config.WAVELET_DWT
        self.level = config.LEVEL_DWT
        self.feature_list = config.FEATURES_LIST
        self.num_mf = config.NUM_MF
        
    def extract_features(self):
        self.feature = utils.generate_feature(self.X,
                            threshold_WAMP=self.threshold_WAMP,
                            threshold_ZC=self.threshold_ZC,
                            threshold_SSC=self.threshold_SSC,
                            bins=self.bins,
                            ranges=self.ranges,
                            num = self.num_mf,
                            wavelet=self.wavelet,
                            level=self.level,
                            show_para=True,
                            feature_list=self.feature_list)
        
        self.feature2 = utils.generate_feature(self.X2,
                            threshold_WAMP=self.threshold_WAMP,
                            threshold_ZC=self.threshold_ZC,
                            threshold_SSC=self.threshold_SSC,
                            bins=self.bins,
                            ranges=self.ranges,
                            num = self.num_mf,
                            wavelet=self.wavelet,
                            level=self.level,
                            show_para=True,
                            feature_list=self.feature_list)
        
        self.feature3 = utils.generate_feature(self.X3,
                            threshold_WAMP=self.threshold_WAMP,
                            threshold_ZC=self.threshold_ZC,
                            threshold_SSC=self.threshold_SSC,
                            bins=self.bins,
                            ranges=self.ranges,
                            num = self.num_mf,
                            wavelet=self.wavelet,
                            level=self.level,
                            show_para=True,
                            feature_list=self.feature_list)

    @property
    def train_set(self):
        return self.feature,self.Y,self.F
    
    @property
    def valid_set(self):
        return self.feature2,self.Y2,self.F2
    
    @property
    def test_set(self):
        return self.feature3,self.Y3,self.F3
    
        