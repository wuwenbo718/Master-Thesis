from scipy.stats import skew
from scipy.sparse import spdiags
import numpy as np
import pandas as pd
from nitime.algorithms.autoregressive import AR_est_YW
import pywt
from scipy import signal
from scipy.integrate import cumtrapz
from scipy.stats import zscore,kurtosis
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf


def detrend(signal, Lambda, return_trend=False):
    """detrend(signal, Lambda) -> filtered_signal
  
    This function applies a detrending filter.
   
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
  
    Inputs:

    signal: The signal which you want to remove the trend.

    Lambda (int): The smoothing parameter.

    Outputs:
  
    filtered_signal: The detrended signal.
    trend: The trend of original signal.
    """
    signal_length = signal.shape[1]
    
    # observation matrix
    H = np.identity(signal_length) 

    # second-order difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2*np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length-2), signal_length).toarray()
    filtered_signal = (H - np.linalg.inv(H + (Lambda**2) * D.T@D))@signal
    if return_trend:
        trend = (np.linalg.inv(H + (Lambda**2) * D.T@D))@signal
        return filtered_signal,trend
    else:
        return filtered_signal
    

    
def rectify_emg(emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
    """
    Compute envelope of EMG signal with lowpass filters
    
    Inputs:
    emg: EMG data
    high: high-pass cut off frequency
    sfreq: sampling frequency
    
    Outputs:
    emg_envelope: envelope of EMG signal
    
    """
    
    # normalise cut-off frequencies to sampling frequency
    low_band = low_band/(sfreq/2)
    
    # create lowpass filter for EMG
    b1, a1 = signal.butter(4, [low_band], btype='lowpass')
    
    # process EMG signal: filter EMG
    emg_filtered = signal.filtfilt(b1, a1, emg)    
    
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/(sfreq/2)
    b2, a2 = signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = signal.filtfilt(b2, a2, emg_rectified)
    return emg_envelope

def mean_smooth(data,neighbor=5):
    """
    Smooth the signal with mean.
    """
    if data.shape[0] == 0:
        print('empty dataset.')
        return np.array([])
    [m,n,l]=data.shape
    temp = np.zeros((m,n+neighbor*2,l))
    temp[:,neighbor:-neighbor,:]=data
    results = np.zeros((m,n,l))
    for i in range(neighbor*2+1):
        results += temp[:,i:n+i,:]
    return results/(neighbor*2+1)

def rectify_emg_moving_average(emg, neighbor=10):
    """
    Compute the envelope of EMG signal with averaging methode.
    
    Inputs:
    emg: EMG data
    neighbor: the number of current point's neighbor for averaging
    
    Outputs:
    Envelope of rectified EMG signal
    """
    
    # process EMG signal: rectify
    emg_rectified = abs(emg)
    
    # apply to rectified signal to get EMG envelope
    emg_envelope = mean_smooth(emg_rectified,neighbor)
    return emg_envelope

def generate_window_slide_data_time_continue(data,width = 256,stride = 64,scaler=False,same_label=False,drop_with_zscore=None,remove_freqs=False):
    """
    Segment the signal. Only keep the segments with continuous time.
    
    Inputs:
    data: the signal to segment
    width: window size
    stride: step size
    scaler: if True use standard scaler on data
    same_label: if True only use the data with same label1 and label2
    
    Outputs:
    segmented signal
    """    

    if same_label:
        ind = (data.Label1 == data.Label2)
        data = data.loc[ind,:].reset_index(drop=True)
#         print(len(data))
        
    if drop_with_zscore:
        
        ind = abs(zscore(data.iloc[:,3:],axis=0)) > drop_with_zscore
        ind = data.index[np.any(ind,axis=1)]
        data = data.drop(ind).reset_index(drop=True)
#         print(len(data))
        
    l = len(data)
    end = (l-width)//stride+1
    X = []
    Y = []
    if scaler:
        sc = StandardScaler(with_mean = True)
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                temp = np.array(data.iloc[i*stride:i*stride+width,3:])
                time = np.array(data.iloc[i*stride:i*stride+width,0])
                if (np.round(time[1:]-time[:-1],3)>0.001).any():
                    continue
                    
                if remove_freqs:
                    freqs, power=signal.periodogram(temp, 1e3,axis=0)
#                     ind_l = freqs>20
                    ind_l = (freqs<250) & (freqs>20)
                    max_l = np.max(power[ind_l],axis=0)
    #                 max_h = np.max(power[~ind_l],axis=0)
                    # The amplitude of frequency components that is lower than 20 Hz must be less than 10 times of the other components and the amplitude of max frequency must over 0.5
#                     if  np.sum(max_l<0.6)>=temp.shape[1]/2:
                    if  np.any(max_l<0.5):
#                         print(np.round(time[0],3),':',np.round(time[-1],3))
                        continue
                    
                Y += [data.Label2[i*stride]]
                x_sc = sc.fit_transform(temp)
                X += [x_sc]
            else:
                continue
    else:
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                temp = np.array(data.iloc[i*stride:i*stride+width,3:])
                time = np.array(data.iloc[i*stride:i*stride+width,0])
                if (np.round(time[1:]-time[:-1],3)>0.001).any():
                    continue
                
                if remove_freqs:
                    freqs, power=signal.periodogram(temp, 1e3,axis=0)
                    ind_l = freqs<250
                    max_l = np.max(power[ind_l],axis=0)
    #                 max_h = np.max(power[~ind_l],axis=0)
                    # The amplitude of frequency components that is lower than 20 Hz must be less than 10 times of the other components and the amplitude of max frequency must over 0.5
                    if  np.any(max_l<0.5):
                        continue
                    
                Y += [data.Label2[i*stride]]
                X += [temp]
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

def compute_CWT_feature(data,scale=32,wavelet = 'mexh'):
    """
    Compute features based on continuous wavelet transform of EMG signal
    """
    n,t,c = data.shape
    f=4
    cwt = np.zeros((n,f*c))
    #print(cwt.shape)
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * scale
    scales = cparam / np.arange(int(scale+1), 1, -1)
    #scales = np.arange(1,scale+1)
    batch = 5
    N = np.ceil(n/batch).astype(np.int32)
    
    for i in range(N):
        cwtmatr,_ = pywt.cwt(data[i*batch:(i+1)*batch,:,:],scales,wavelet,axis=1)
        cwtmatr = cwtmatr.transpose(1,0,2,3)
        mean_abs = np.mean(np.abs(cwtmatr),axis=2)
        mean_coe = np.mean(mean_abs,axis=1)
        min_coe = np.min(mean_abs,axis=1)
        mean_scale = scales@mean_abs/mean_abs.sum(axis=1)
        total = (cumtrapz(mean_abs,scales,axis=1,initial=0))
        w=np.greater_equal(total,total[:,-1:,:]/2)
        median_scale = np.zeros((cwtmatr.shape[0],c))
        for j in range(cwtmatr.shape[0]):
            for k in range(c):
                median_scale[j,k] = scales[w[j,:,k]][0]

        cwt[i*batch:(i+1)*batch,::f] = mean_coe
        cwt[i*batch:(i+1)*batch,1::f] = min_coe
        cwt[i*batch:(i+1)*batch,2::f] = mean_scale
        cwt[i*batch:(i+1)*batch,3::f] = median_scale
        
    return cwt

def compute_IEMG(data):
    """
    Compute integrated EMG
    """
    IEMG = np.sum(np.abs(data),axis=1)
    return IEMG

def compute_MAV(data):
    """
    Compute mean average value of data
    """
    N = data.shape[1]
    return np.sum(np.abs(data),axis=1)/N

def compute_SSI(data):
    """
    Compute simple square data
    """
    return np.sum(np.power(data,2),axis=1)

def compute_VAR(data):
    """
    Compute variance of data
    """
    N = data.shape[1]
    return compute_SSI(data)/(N-1)

def compute_RMS(data):
    """
    Compute root mean square of EMG data
    """
    N = data.shape[1]
    return np.sqrt(compute_SSI(data)/N)

def compute_WL(data):
    """
    Compute waveform length
    """
    temp = data[:,1:,:]-data[:,:-1,:]
    return compute_IEMG(temp)

def compute_ZC(data,threshold=0):
    """
    Compute zero crossing
    data: EMG signal
    threshold: Threshold condition is used to avoid from background noise.
    
    outputs:
    the number of times that the amplitude values of EMG signal crosses zero in x-axis.
    """
    l = len(data)
    sign = ((data[:,1:,:])*(data[:,:-1,:]))<0
    sub = np.abs(data[:,1:,:]-data[:,:-1,:])>threshold
    return np.sum(sign & sub,axis=1)/l

def compute_ku(data):
    """
    Compute kurtosis of data
    """
    return kurtosis(data,1)

def compute_SSC(data,threshold=0):
    """
    Compute slope sign change
    """
    sign = (data[:,1:-1,:]-data[:,:-2,:])*(data[:,2:,:]-data[:,1:-1,:])
    ssc = (sign > 0) & (((data[:,1:-1,:]-data[:,:-2,:])>threshold) | ((data[:,1:-1,:]-data[:,2:,:])>threshold))
    return np.sum(ssc,axis=1)

def compute_WAMP(data,threshold = 0):
    """
    Compute the number of time resulting from the difference between EMG signal amplitude of two adjoining segments that exceeds a predefined threshold
    """
    temp = np.abs(data[:,1:,:]-data[:,:-1,:])>=threshold
    return np.sum(temp,axis=1)

def compute_Skewness(data):
    """
    Compute skewness of data
    """
    return skew(data,axis=1)

def compute_Acti(data):
    """
    Compute activity which is one of Hjorth Parameters
    """
    N = data.shape[1]
    mean = np.mean(data,axis=1)
    return np.sum((data-mean[:,np.newaxis,:])**2,axis=1)/N

def compute_Mobi(data):
    """
    Compute mobility which is one of Hjorth Parameters
    """
    N,L,C = data.shape
    feature = np.zeros((N,C))
    for i in range(N):
        for j in range(C):
            temp = np.gradient(data[i,:,j],np.arange(0,L/1000,1e-3))
            feature[i,j] = np.sum((temp-temp.mean())**2)/N
    acti = compute_Acti(data)
    feature = np.sqrt(feature/acti)
    return feature

def compute_complexity(data):
    """
    Compute complexity which is one of Hjorth Parameters
    """
    N,L,C = data.shape
    xd = np.zeros((N,L,C))
    for i in range(N):
        for j in range(C):
            xd[i,:,j] = np.gradient(data[i,:,j],np.arange(0,L/1000,1e-3))
    return compute_Mobi(xd)/compute_Mobi(data)

def compute_AR(data,p=4):
    """
    Compute Autoregression Coefficient
    
    Inputs:
    data: EMG signal
    p: Model order
    
    Outputs:
    Autoregression Coefficient
    """
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,j] = ak[0]
    return feature

def compute_cc(ak,p=4):
    '''
    Compute  Cepstral Coefficient with Autoregression Coefficient for compute_CC
    
    Inputs:  
    ak: Autoregression Coefficient
    p: Model order
    
    Outputs: Cepstral Coefficient
    '''
    cc = np.zeros(p)
    for i in np.arange(p):
        temp = -ak[i]
        for j in range(i):
            temp -= (1-(j+1)/(i+1))*ak[j]*cc[i-j-1]
        cc[i]=temp
    return cc

def compute_CC(data,p=4):
    
    '''
    Compute Cepstral Coefficient of data matrix
    
    Inputs: 
    data [N,L,C]
    N: number of data
    L: length of signal
    C: number of channels 
    
    p: Model order
    
    Outputs: Cepstral Coefficient
    '''
    
    N = len(data)
    C = data.shape[-1]
    feature = np.zeros((N,p,C))
    
    for i in range(N):
        for j in range(C):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,:,j] = compute_cc(ak,p)
            
    return feature

def compute_HIST(data,bins=9,ranges=(-10,10)):
    """
    Compute EMG Histogram
    
    Inputs:
    data: EMG signal
    bins: the number of bins
    ranges: the lower and upper range of the bins
    
    Outputs:
    Histogram
    """
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,bins*M))
    
    for i in range(N):
        for j in range(M):
            hist,_ = np.histogram(data[i,:,j],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
            
    return feature

def compute_MaxFreq(data,num=3):
    """
    Compute max frequency of EMG signal (pandas output version)
    
    Inputs:
    data: EMG signal
    num: Take the max num frequency to calculate the mean value as max frequency
    
    Outputs:
    Max frequency
    """
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,M))

    freqs, power=signal.periodogram(data, 1e3, axis = 1)
    
    for i in range(N):
        for j in range(M):
            ind = np.argsort(-power[i,:,j])[:num]
            feature[i,j] = np.mean(freqs[ind])
            
    return feature

def compute_MDF(data):
    """
    Compute median frequency 
    """
    N,M = data.shape[0::2]
    feature = np.zeros((N,M))
    
    freqs, power=signal.periodogram(data, 1e3,axis=1)
    total = (cumtrapz(power,freqs,axis=1,initial=0))
    w=np.greater_equal(total,total[:,-1:,:]/2)
    
    for i in range(N):
        for j in range(M):
            feature[i,j] = freqs[w[i,:,j]][0]
            
    return feature

def compute_MNF(data):
    """
    Compute mean frequency
    """
    freqs, power=signal.periodogram(data, 1e3, axis=1)
    feature = freqs@power/power.sum(axis=1)
    return feature

def compute_mDWT(data,wavelet='db7',level=3):
    """
    Compute Marginal Discrete Wavelet Transform
    """
    N,M = data.shape[0::2]
    wa = pywt.wavedec(data,wavelet,level=level,axis=1)
    wa = [w for w in wa]
    wa = np.concatenate(wa,axis=1)
    L = wa.shape[1]
    S = int(np.log2(L))
    features = np.zeros((N,S*M))
    for i in range(S):
        C = L//(2**(i+1))-1
        features[:,i::S] = np.abs(wa[:,:C+1,:]).sum(axis=1)
    return features

def generate_feature(data,threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     num = 3,
                     wavelet='db7',
                     level=3,
                     show_para=True,
                    feature_list=['IEMG','SSI','WL','ZC','ku','SSC','skew','Acti','AR','HIST','MDF','MNF','mDWT']
                    ):
    """
    Generate features.
    Inputs:
    data: (N,L,C)
    N: Number of signal
    L: Length of signal
    C: Number of channels
    
    threshold_WAP: threshold for willison amplitude
    threshold_ZC: threshold for zero crossing
    bins: bins for EMG Histogram
    ranges: ranges fof EMG Histogram
    show_para: if True show the list of name of features
    """
    feature = []
    if 'IEMG' in feature_list:
        IEMG = compute_IEMG(data)
        feature+=[IEMG]
    if 'MAV' in feature_list:
        MAV = compute_MAV(data)
        feature+=[MAV]
    if 'SSI' in feature_list:
        SSI = compute_SSI(data)
        feature+=[SSI]
    if 'VAR' in feature_list:
        VAR = compute_VAR(data)
        feature+=[VAR]
    if 'RMS' in feature_list:
        RMS = compute_RMS(data)
        feature+=[RMS]
    if 'WL' in feature_list:
        WL = compute_WL(data)
        feature+=[WL]
    if 'ZC' in feature_list:
        ZC = compute_ZC(data,threshold_ZC)
        feature+=[ZC]
    if 'ku' in feature_list:
        ku = compute_ku(data)
        feature+=[ku]
    if 'SSC' in feature_list:
        SSC = compute_SSC(data,threshold_SSC)
        feature+=[SSC]
    if 'WAMP' in feature_list:
        WAMP = compute_WAMP(data,threshold_WAMP)
        feature+=[WAMP]
    if 'skew' in feature_list:
        skew = compute_Skewness(data)
        feature+=[skew]
    if 'Acti' in feature_list:
        Acti = compute_Acti(data)
        feature+=[Acti]
    if 'Mobi' in feature_list:
        Mobi = compute_Mobi(data)
        feature+=[Mobi]
    if 'Comp' in feature_list:
        Comp = compute_complexity(data)
        feature+=[Comp]
    if 'AR' in feature_list:
        AR = compute_AR(data)
        feature+=[AR]
    if 'CC' in feature_list:
        CC = compute_CC(data)
        feature+=[CC]
    if 'HIST' in feature_list:
        HIST = compute_HIST(data,bins=bins,ranges=ranges)
        feature+=[HIST]
    if 'MF' in feature_list:
        MF = compute_MaxFreq(data,num=num)
    if 'MDF' in feature_list:
        MDF = compute_MDF(data)
        feature+=[MDF]
    if 'MNF' in feature_list:
        MNF = compute_MNF(data)
        feature+=[MNF]
    if 'mDWT' in feature_list:
        mDWT = compute_mDWT(data,wavelet,level)
        feature+=[mDWT]
    
    feature = np.concatenate(feature,axis =1)
    if show_para:
        print('threshold_WAMP:%0.1f, threshold_ZC:%0.1f, threshold_SSC:%0.1f, bins:%d, ranges:(%d,%d), num_mf:%d, wavelet: %s, level: %d'
          %(threshold_WAMP,threshold_ZC,threshold_SSC,bins,ranges[0],ranges[1],num, wavelet, level))
        print(feature_list)
    return feature

def data_split_oh(data,labels,
                  class_id=None,
                  binary=True,
                  test_size=0.25,
                  shuffle=True,
                  random_state=None):
    """
    Split data and labels, process them for Model
    Inputs:
    data: data to process
    labels: labels of data
    class_id: class to classify
    binary: if True do classification of 0 and others, 
                if False do classification of class_id
    shuffle: if True shuffle the data for train- and validset split
    random_state: random seed for shuffle
                
    Outputs:
    Train-,valid-,test data and labels(one-hot form)
    
    """
    if binary:
        inds = []
        inds_1 = []
        for l in labels:
            inds += [((l==0)|(l==1)|(l==2)|(l==3)|(l==4)|(l==6))]
            inds_1 += [l!=0]
        
        label_n = []
        for i,l in zip(inds,labels):
            label_n += [l[i].copy()]

        for j,ind in enumerate(inds_1):
            label_n[j][ind] = 1
 
        oh = OneHotEncoder()
        oh.fit(label_n[0][:,np.newaxis])

    else:
        if class_id == None:
            class_id = set(labels[0])
            
        inds = []
        for l in labels:
            temp = np.ones(len(l)) == 0
            for c in class_id:
                temp |= l == c
            inds += [temp]

        label_n = []
        for i,l in zip(inds,labels):
            label_n += [l[i].copy()]

        oh = OneHotEncoder()
        oh.fit(label_n[0][:,np.newaxis])

    x_train = data[0][inds[0]]
    y_train = oh.transform(label_n[0][:,np.newaxis]).toarray()
    x_valid = data[1][inds[1]]
    y_valid = oh.transform(label_n[1][:,np.newaxis]).toarray()
    
    x_train,x_valid,y_train,y_valid = train_test_split(np.concatenate([x_train,x_valid]),
                                      np.concatenate([y_train,y_valid]),
                                      test_size=test_size,
                                      random_state=random_state,
                                      shuffle=shuffle)
    
    x_test = data[2][inds[2]]
    y_test = oh.transform(label_n[2][:,np.newaxis]).toarray()
    
    return x_train,y_train,x_valid,y_valid,x_test,y_test

def data_split(data,labels,
               class_id=None,
               binary=True,
               test_size=0.25,
               shuffle=True,
               random_state=None):
    """
    Split data and labels, process them for Model
    Inputs:
    data: data to process
    labels: labels of data
    class_id: class to classify
    binary: if True do classification of 0 and others, 
                if False do classification of class_id
    shuffle: if True shuffle the data for train- and validset split
    random_state: random seed for shuffle
                
    Outputs:
    Train-,valid-,test data and labels
    
    """
    if binary:
        inds = []
        inds_1 = []
        for l in labels:
            inds += [((l==0)|(l==1)|(l==2)|(l==3)|(l==4)|(l==6))]
            inds_1 += [l!=0]
        
        label_n = []
        for i,l in zip(inds,labels):
            label_n += [l[i].copy()]

        for j,ind in enumerate(inds_1):
            label_n[j][ind] = 1

    else:
        if class_id == None:
            class_id = set(labels[0])
            
        inds = []
        for l in labels:
            temp = np.ones(len(l)) == 0
            for c in class_id:
                temp |= l == c
            inds += [temp]

        label_n = []
        for i,l in zip(inds,labels):
            label_n += [l[i].copy()]

    x_train = data[0][inds[0]]
    y_train = label_n[0]
    x_valid = data[1][inds[1]]
    y_valid = label_n[1]
    
    x_train,x_valid,y_train,y_valid = train_test_split(np.concatenate([x_train,x_valid]),
                                      np.concatenate([y_train,y_valid]),
                                      test_size=test_size,
                                      random_state=random_state,
                                      shuffle=shuffle)
    
    x_test = data[2][inds[2]]
    y_test = label_n[2]
    
    return x_train,y_train,x_valid,y_valid,x_test,y_test

def cwt_tf(data, scales, wavelet, batch_size, sampling_period=1.):

    # accept array_like input; make a copy to ensure a contiguous array
    dt = tf.float32
    dt_cplx = tf.complex64
    if not isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
        wavelet = pywt.DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.array([scales])

    dt_out = dt_cplx if wavelet.complex_cwt else dt
    out = []
    precision = 10
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi

    # convert int_psi, x to the same precision as the data
    dt_psi = np.complex64 if int_psi.dtype.kind == 'c' else np.float32
    int_psi = np.asarray(int_psi, dtype=dt_psi)
    x = np.asarray(x, dtype=np.float32)

    for i, scale in enumerate(scales):
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]
        int_psi_scale_n = np.zeros((len(int_psi_scale),data.shape[-1],data.shape[-1]),dtype=int_psi_scale.dtype)
        for c in range(data.shape[-1]):
            int_psi_scale_n[:,c,c] = int_psi_scale
       
        conv = tf.nn.convolution(data, int_psi_scale_n, 1, 'SAME')
    
        coef = - tf.math.sqrt(float(scale)) * tf.experimental.numpy.diff(conv, axis=-2)

        if dt_out != dt_cplx:
            coef = tf.math.real(coef)

        out += [tf.cast(tf.abs(coef),dt_out)]
        
#     frequencies = pywt.scale2frequency(wavelet, scales, precision)
#     if np.isscalar(frequencies):
#         frequencies = np.array([frequencies])
#     frequencies /= sampling_period
    return tf.transpose(tf.convert_to_tensor(out,dtype=dt_out),(1,0,2,3))#, frequencies