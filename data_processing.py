from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import numpy as np
import pandas as pd
from nitime.algorithms.autoregressive import AR_est_YW

def scale_data(data,scaler):
    X = data.iloc[:,3:]
    X = sc.fit_transform(X)
    data.iloc[:,3:] = X
    return data

def generate_window_slide_data(data,width = 256,stride = 32,scaler=False):
    l = len(data)
    end = (l-width)//stride+1
    X = []
    Y = []
    if scaler:
        sc = StandardScaler(with_mean = False)
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                Y += [data.Label2[i*stride]]
                x_sc = sc.fit_transform(np.array(data.iloc[i*stride:i*stride+width,3:]))
                X += [x_sc]
                #print(set(data.Label2[i*stride:i*stride+width]))
            else:
                continue
    else:
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                Y += [data.Label2[i*stride]]
                X += [np.array(data.iloc[i*stride:i*stride+width,3:])]
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

#def generate_CWT_feature(data,widths=260,wavelet = signal.ricker):
#    n,t,c = data.shape
#    cwtmatr = np.zeros((n,widths,t,c))
#    for i in range(n):
#        for j in range(c):
#            cwtmatr[i,:,:,j] = signal.cwt(data[i,:,j],wavelet,np.arange(1,widths+1))
#    return cwtmatr


def generate_CWT_feature(data,widths=260,wavelet = 'mexh'):
    n,t,c = data.shape
    cwtmatr = np.zeros((n,widths,t,c))
    for i in range(n):
        for j in range(c):
            cwtmatr[i,:,:,j],_ = pywt.cwt(data[i,:,j],np.arange(1,widths+1),wavelet)
    return cwtmatr

def compute_IEMG(data):
    IEMG = np.sum(np.abs(data),axis=1)
    return IEMG

def compute_MAV(data):
    N = data.shape[1]
    return np.sum(np.abs(data),axis=1)/N

def compute_SSI(data):
    return np.sum(np.power(data,2),axis=1)

def compute_VAR(data):
    N = data.shape[1]
    return compute_SSI(data)/(N-1)

def compute_RMS(data):
    N = data.shape[1]
    return np.sqrt(compute_SSI(data)/N)

def compute_WL(data):
    temp = data[:,1:,:]-data[:,:-1,:]
    return compute_IEMG(temp)

def compute_ZC(data,threshold=0):
    noise = 1e-2
    data = data+noise
    sign = ((data[:,1:,:])*(data[:,:-1,:]))<=-threshold
    sub = np.abs(data[:,1:,:]-data[:,:-1,:])>threshold
    return compute_IEMG(sign & sub)

def compute_SSC(data,threshold=0):
    temp = (data[:,1:-1,:]-data[:,:-2,:])*(data[:,1:-1,:]-data[:,2:,:])
    return compute_IEMG(temp >= threshold)

def compute_WAMP(data,threshold = 0):
    temp = np.abs(data[:,1:,:]-data[:,:-1,:])>=threshold
    return np.sum(temp,axis=1)

def compute_Skewness(data):
    return skew(data,axis=1)

def compute_Acti(data):
    N = data.shape[1]
    mean = np.mean(data,axis=1)
    return np.sum((data-mean[:,np.newaxis,:])**2,axis=1)/N

def compute_AR(data,p=4):
    N = len(data)
    feature = np.zeros((N,8))
    for i in range(N):
        for j in range(8):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,j] = ak[0]
    return feature

def compute_HIST(data,bins=9,ranges=(-10,10)):
    N = len(data)
    feature = np.zeros((N,bins*8))
    for i in range(N):
        for j in range(8):
            hist,_ = np.histogram(data[i,:,j],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
    return feature

def generate_feature(data,threshold_WAMP=30,threshold_ZC=0,threshold_SSC=0,bins=9,ranges=(-10,10)):
    IEMG = compute_IEMG(data)
    MAV = compute_MAV(data)
    SSI = compute_SSI(data)
    VAR = compute_VAR(data)
    RMS = compute_RMS(data)
    WL = compute_WL(data)
    ZC = compute_ZC(data,threshold_ZC)
    SSC = compute_SSC(data,threshold_SSC)
    WAMP = compute_WAMP(data,threshold_WAMP)
    skew = compute_Skewness(data)
    Acti = compute_Acti(data)
    AR = compute_AR(data)
    HIST = compute_HIST(data,bins=bins,ranges=ranges)
    feature = np.concatenate([IEMG,MAV,SSI,VAR,RMS,WL,ZC,SSC,WAMP,skew,Acti,AR,HIST],axis =1)
    print('threshold_WAMP:%0.1f, threshold_ZC:%0.1f, threshold_SSC:%0.1f,bins:%d,ranges:(%d,%d)'
          %(threshold_WAMP,threshold_ZC,threshold_SSC,bins,ranges[0],ranges[1]))
    print('IEMG,MAV,SSI,VAR,RMS,WL,ZC,SSC,WAMP,skew,Acti,AR,HIST')
    return feature

def pipeline_feature(path,width = 256,stride = 32,scaler=False,threshold_WAMP=30,threshold_ZC=0,threshold_SSC=0,bins=9,ranges=(-10,10)):
    emg_data = pd.read_csv(path)
    emg_data = emg_data.fillna({'LEFT_TA':emg_data.LEFT_TA.mean(),
                           'LEFT_TS':emg_data.LEFT_TS.mean(),
                           'LEFT_BF':emg_data.LEFT_BF.mean(),
                           'LEFT_RF':emg_data.LEFT_RF.mean(),
                           'RIGHT_TA':emg_data.RIGHT_TA.mean(),
                           'RIGHT_TS':emg_data.RIGHT_TS.mean(),
                           'RIGHT_BF':emg_data.RIGHT_BF.mean(),
                           'RIGHT_RF':emg_data.RIGHT_RF.mean()})
    x,y = generate_window_slide_data(emg_data,width=width,stride=stride,scaler=scaler)
    feature = generate_feature(x,threshold_WAMP=threshold_WAMP,threshold_ZC=threshold_ZC,threshold_SSC=threshold_SSC,bins=bins,ranges=ranges)
    return feature,y