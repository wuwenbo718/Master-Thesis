from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
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

#def scale_data(data,scaler):
#    X = data.iloc[:,3:]
#    X = sc.fit_transform(X)
#    data.iloc[:,3:] = X
#    return data

def scale_data(data,sc,cwt=True):
    M,N,I,J = data.shape
    result = np.zeros((M,N,I,J))
    #sc = StandardScaler()
    if cwt:
        for i in range(M):
            for j in range(J):
                cwtmatr = data[i,:,:,j]
                result[i,:,:,j] = sc.fit_transform(cwtmatr)
                
        return result
    else:
        X = data.iloc[:,3:]
        X = sc.fit_transform(X)
        data.iloc[:,3:] = X
        return data
    
def detrend(signal, Lambda, return_trend=False):
    """detrend(signal, Lambda) -> filtered_signal
  
    This function applies a detrending filter.
   
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
  
    **Parameters**

    ``signal`` (1d numpy array):
    The signal where you want to remove the trend.

    ``Lambda`` (int):
    The smoothing parameter.

    **Returns**
  
    ``filtered_signal`` (1d numpy array):
    The detrended signal.
    """
    signal_length = signal.shape[0]

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

def lowpass_filter(data):
    x = np.zeros(data.shape)
    N,M = data.shape[0::2]
    wn=2*fn/1000
    b, a = signal.butter(4, wn, 'lowpass')
    for i in range(N):
        for j in range(M):
            x[i,:,j] = signal.filtfilt(b, a, data[i,:,j])
    return x

def bandpass_filter(data,fn=350):
    x = np.zeros(data.shape)
    N,M = data.shape[0::2]
    #wn=2*fn/1000
    fn = 10
    wn=2*fn/1000
    fn1 = 350
    wn1=2*fn1/1000
    b, a = signal.butter(4, [wn,wn1], 'bandpass')
    #b, a = signal.butter(4, wn, 'lowpass')
    for i in range(N):
        for j in range(M):
            x[i,:,j] = signal.filtfilt(b, a, data[i,:,j])
    return x

def mean_smooth(data,neighbor=5):
    [m,n,l]=data.shape
    temp = np.zeros((m,n+neighbor*2,l))
    temp[:,neighbor:-neighbor,:]=data
    results = np.zeros((m,n,l))
    for i in range(neighbor*2+1):
        results += temp[:,i:n+i,:]
    return results/(neighbor*2+1)

def generate_window_slide_data(data,width = 256,stride = 64,scaler=False,same_label=False):
    #sc = joblib.load('./model/scalar')
    #sc = MinMaxScaler()
    if same_label:
        ind = (data.Label1 == data.Label2)
        data = data.loc[ind,:].reset_index(drop=True)
        
    l = len(data)
    end = (l-width)//stride+1
    X = []
    Y = []
    if scaler:
        sc = StandardScaler(with_mean = True)
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                Y += [data.Label2[i*stride]]
                x_sc = sc.fit_transform(np.array(data.iloc[i*stride:i*stride+width,3:]))
                #x_sc = normalize(np.array(data.iloc[i*stride:i*stride+width,3:]),axis=0)
                X += [x_sc]
                #print(set(data.Label2[i*stride:i*stride+width]))
            else:
                continue
    else:
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                Y += [data.Label2[i*stride]]
                #X += [np.clip(np.array(data.iloc[i*stride:i*stride+width,3:]),-500,500)]
                X += [np.array(data.iloc[i*stride:i*stride+width,3:])]
                #print(np.array(data.iloc[i*stride:i*stride+width,3:]).shape)
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

def generate_window_slide_data_mix(data,width = 256,stride = 64,scaler=False,same_label=False,mix=0.8):
    #sc = joblib.load('./model/scalar')
    #sc = MinMaxScaler()
    if same_label:
        ind = (data.Label1 == data.Label2)
        data = data.loc[ind,:].reset_index(drop=True)
        
    l = len(data)
    end = (l-width)//stride+1
    X = []
    Y = []
    if scaler:
        sc = StandardScaler(with_mean = False)
        for i in range(end):
            temp = data.Label2[i*stride:i*stride+width].value_counts()
            label_num = temp.max()
            if label_num/width>mix:
                Y += [temp.index[temp.argmax()]]
                x_sc = sc.fit_transform(np.array(data.iloc[i*stride:i*stride+width,3:]))
                #x_sc = normalize(np.array(data.iloc[i*stride:i*stride+width,3:]),axis=0)
                X += [x_sc]
                #print(set(data.Label2[i*stride:i*stride+width]))
            else:
                continue
    else:
        for i in range(end):
            temp = data.Label2[i*stride:i*stride+width].value_counts()
            label_num = temp.max()
            if label_num/width>mix:
                Y += [temp.index[temp.argmax()]]
                #X += [np.clip(np.array(data.iloc[i*stride:i*stride+width,3:]),-500,500)]
                X += [np.array(data.iloc[i*stride:i*stride+width,3:])]
                #print(np.array(data.iloc[i*stride:i*stride+width,3:]).shape)
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

def generate_window_slide_data_NA_remove(data,width = 256,stride = 64,scaler=False,same_label=False):
    #sc = joblib.load('./model/scalar')
    #sc = MinMaxScaler()
    if same_label:
        ind = (data.Label1 == data.Label2)
        data = data.loc[ind,:].reset_index(drop=True)
        
    l = len(data)
    end = (l-width)//stride+1
    X = []
    Y = []
    if scaler:
        sc = StandardScaler(with_mean = False)
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                temp = np.array(data.iloc[i*stride:i*stride+width,3:])
                if np.isnan(np.min(temp)):
                    continue
                Y += [data.Label2[i*stride]]
                #x_sc = sc.fit_transform(temp)
                x_sc = normalize(temp,axis=0)
                X += [x_sc]
                #print(set(data.Label2[i*stride:i*stride+width]))
            else:
                continue
    else:
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                temp = np.array(data.iloc[i*stride:i*stride+width,3:])
                if np.isnan(np.min(temp)):
                    continue
                Y += [data.Label2[i*stride]]
                #X += [np.clip(np.array(data.iloc[i*stride:i*stride+width,3:]),-500,500)]
                X += [temp]
                #print(np.array(data.iloc[i*stride:i*stride+width,3:]).shape)
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

def generate_window_slide_data_time_continue(data,width = 256,stride = 64,scaler=False,same_label=False):
    #sc = joblib.load('./model/scalar')
    #sc = MinMaxScaler()
    if same_label:
        ind = (data.Label1 == data.Label2)
        data = data.loc[ind,:].reset_index(drop=True)
        
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
                Y += [data.Label2[i*stride]]
                x_sc = sc.fit_transform(temp)
                #x_sc = normalize(temp,axis=0)
                X += [x_sc]
                #print(set(data.Label2[i*stride:i*stride+width]))
            else:
                continue
    else:
        for i in range(end):
            if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                temp = np.array(data.iloc[i*stride:i*stride+width,3:])
                time = np.array(data.iloc[i*stride:i*stride+width,0])
                if (np.round(time[1:]-time[:-1],3)>0.001).any():
                    continue
                Y += [data.Label2[i*stride]]
                #X += [np.clip(np.array(data.iloc[i*stride:i*stride+width,3:]),-500,500)]
                X += [temp]
                #print(np.array(data.iloc[i*stride:i*stride+width,3:]).shape)
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

def generate_window_slide_data2(emg_data,width = 256,stride = 64,same_label=False):
    fn = 20
    wn=2*fn/1000
    fn1 = 300
    wn1 = 2*fn1/1000
    b, a = signal.butter(4, [wn,wn1], 'bandpass')
    fs = 1000.0  # Sample frequency (Hz)
    f0 = 50  # Frequency to be removed from signal (Hz)
    Q = 100.0  # Quality factor
    b1, a1 = signal.iirnotch(f0, Q, fs)
    b2, a2 = signal.iirnotch(75.,Q, fs)
    #b, a = signal.butter(4, wn, filt[1])
    for i in ['LEFT_TA','LEFT_TS','LEFT_BF','LEFT_RF','RIGHT_TA','RIGHT_TS','RIGHT_BF','RIGHT_RF']:
        emg_data.loc[:,i] = signal.filtfilt(b, a, emg_data.loc[:,i])
        emg_data.loc[:,i] = signal.filtfilt(b1, a1, emg_data.loc[:,i])
        emg_data.loc[:,i] = signal.filtfilt(b2, a2, emg_data.loc[:,i])
        ind = abs(zscore(emg_data.loc[:,i]))<10
        emg_data.loc[~ind,i]=emg_data.loc[ind,i].mean()
    sc = StandardScaler()
    emg_data.iloc[:,3:] = sc.fit_transform(emg_data.iloc[:,3:])
    x,y = generate_window_slide_data(emg_data,width = width,stride = stride,scaler=False,same_label=same_label)
    
    return x,y

#def generate_CWT_feature(data,widths=260,wavelet = signal.ricker):
#    n,t,c = data.shape
#    cwtmatr = np.zeros((n,widths,t,c))
#    for i in range(n):
#        for j in range(c):
#            cwtmatr[i,:,:,j] = signal.cwt(data[i,:,j],wavelet,np.arange(1,widths+1))
#    return cwtmatr


def generate_CWT_feature(data,scale=32,wavelet = 'mexh'):
    n,t,c = data.shape
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * scale
    scales = cparam / np.arange(int(scale+1), 1, -1)
    cwtmatr = np.zeros((n,scale,t,c))
    for i in range(n):
        for j in range(c):
            temp,_ = pywt.cwt(data[i,:,j],scales,wavelet)
            cwtmatr[i,:,:,j] = abs(temp)
    return cwtmatr

def compute_CWT_feature(data,scale=32,wavelet = 'mexh'):
    n,t,c = data.shape
    cwt = np.zeros((n,4*c))
    #print(cwt.shape)
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * scale
    scales = cparam / np.arange(int(scale+1), 1, -1)
    #scales = np.arange(1,scale+1)
    for i in range(n):
        for j in range(c):
            cwtmatr,_ = pywt.cwt(data[i,:,j],scales,wavelet)
            mean_abs = np.mean(np.abs(cwtmatr),axis=1)
            mean_coe = np.mean(mean_abs)
            min_coe = np.min(mean_abs)
            mean_scale = mean_abs@scales/mean_abs.sum()
            total = (cumtrapz(mean_abs,scales))
            #print(i,j,total[-1])
            w=np.where(total>=(total[-1]/2))[0][0]
            median_scale = w
            #print(i,j*4,(j+1)*4)
            cwt[i,j*4:(j+1)*4] = [mean_coe,min_coe,mean_scale,median_scale]
    return cwt

def standard(x):
    results = np.array([])
    for i in x:
        xs = np.array(i).astype(float)
        xs -= np.mean(i)
        xs /= np.std(i)
        results = np.concatenate([results,xs])
    return results

def compute_DWT(data,wavelet='db7',level=3):
    N,M = data.shape[0::2]
    feature = []
    sc = StandardScaler()
    for i in range(N):
        temp = []
        for j in range(M):
            wa = pywt.wavedec(data[i,:,j],wavelet,level)
            #wa = np.concatenate(wa)
            wa = standard(wa)
            temp.extend(wa)
        feature.append(temp)
    return feature    

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
    #noise = 1e-2
    #data = data+noise
    l = len(data)
    sign = ((data[:,1:,:])*(data[:,:-1,:]))<0
    sub = np.abs(data[:,1:,:]-data[:,:-1,:])>threshold
    return np.sum(sign & sub,axis=1)/l

def compute_ku(data):
    return kurtosis(data,1)

def compute_ZC_expand(data,threshold):
    #noise = 1e-2
    #data = data+noise
    n,_,c=data.shape
    m = len(threshold)
    results = np.zeros((n,m,c))
    sign = ((data[:,1:,:])*(data[:,:-1,:]))<0
    for i in range(m):
        sub = np.sign(threshold[i]+1e-5)*(data[:,1:,:]-data[:,:-1,:])>np.abs(threshold[i])
        results[:,i,:]=np.sum(sign & sub,axis=1)
    return results

def compute_ZC_expand_pd(data,threshold):

    n,_,c=data.shape
    m = len(threshold)
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for j in range(m):
        columns_b += ['_ZC%d'%j]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    results = np.zeros((n,m,c))
    sign = ((data[:,1:,:])*(data[:,:-1,:]))<0
    for i in range(m):
        sub = np.sign(threshold[i]+1e-5)*(data[:,1:,:]-data[:,:-1,:])>np.abs(threshold[i])
        results[:,i,:]=np.sum(sign & sub,axis=1)
    return pd.DataFrame(results.reshape((-1,m*c)),columns=index)

def compute_SSC(data,threshold=0):
    sign = (data[:,1:-1,:]-data[:,:-2,:])*(data[:,2:,:]-data[:,1:-1,:])
    ssc = (sign > 0) & (((data[:,1:-1,:]-data[:,:-2,:])>threshold) | ((data[:,1:-1,:]-data[:,2:,:])>threshold))
    return np.sum(ssc,axis=1)

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
    M = data.shape[-1]
    feature = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,j] = ak[0]
    return feature

def compute_AR_pd(data,p=4):
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,M*p))
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for m in range(p):
        columns_b += ['_AR%d'%m]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    for i in range(N):
        for j in range(M):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,j*p:(j+1)*p] = ak
    return pd.DataFrame(feature,columns=index)

def compute_HIST(data,bins=9,ranges=(-10,10)):
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,bins*M))
    for i in range(N):
        for j in range(M):
            hist,_ = np.histogram(data[i,:,j],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
    return feature

def compute_HIST_pd(data,bins=9,ranges=(-10,10)):
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,bins*M))
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for m in range(bins):
        columns_b += ['_HIST%d'%m]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    #print(index)
    for i in range(N):
        for j in range(M):
            hist,_ = np.histogram(data[i,:,j],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
    return pd.DataFrame(feature,columns=index)

def compute_FHIST_pd(data,bins=5,ranges=(0,300),threshold = 0.5):
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,bins*M))
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for m in range(bins):
        columns_b += ['_FHIST%d'%m]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    #print(index)
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            #sc = MinMaxScaler((0,1))
            #power = sc.fit_transform(power[:,np.newaxis])
            ind = (power > threshold)#[:,0]
            hist,_ = np.histogram(freqs[ind],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
    return pd.DataFrame(feature,columns=index)    

def compute_MaxFreq_pd(data,num=3):
    N = len(data)
    M = data.shape[-1]
    feature = np.zeros((N,num*M))
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for m in range(num):
        columns_b += ['_MF%d'%m]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    #print(index)
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            #sc = MinMaxScaler((0,1))
            #power = sc.fit_transform(power[:,np.newaxis])
            ind = np.argsort(-power)[:num]
            feature[i,j*num:(j+1)*num] = freqs[ind]
    return pd.DataFrame(feature,columns=index)  

def compute_MDF(data):
    N,M = data.shape[0::2]
    feature = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            total = (cumtrapz(power,freqs))
            #print(i,j,np.where(total>=(total[-1]/2)))
            w=np.where(total>=(total[-1]/2))[0][0]
            feature[i,j] = freqs[w]
            #print(w,freqs[w])
    return feature

def compute_MNF(data):
    N,M = data.shape[0::2]
    feature = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            feature[i,j] = freqs@power/power.sum()
    return feature

def mDWT(data,wavelet='db7',level=3):
    wa = pywt.wavedec(data,wavelet,level=level)
    #print(len(wa))
    wa = np.concatenate(wa)
    N = len(wa)
    S = int(np.log2(N))
    M = []
    for i in range(S):
        C = N//(2**(i+1))-1
        #print(C)
        M.append(np.abs(wa[:C+1]).sum())
    return M
    
def compute_mDWT(data,wavelet='db7',level=3):
    N,M = data.shape[0::2]
    feature = []
    for i in range(N):
        temp = []
        for j in range(M):
            temp.extend(mDWT(data[i,:,j],wavelet,level))
        feature.append(temp)
    return feature

def compute_mDWT_pd(data,wavelet='db7',level=3):
    N,M = data.shape[0::2]
    feature = []
    for i in range(N):
        temp = []
        for j in range(M):
            temp.extend(mDWT(data[i,:,j],wavelet,level))
        feature.append(temp)
        
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for m in range(len(feature[0])//8):
        columns_b += ['_mDWT%d'%m]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
        
    return pd.DataFrame(feature,columns=index)

def generate_feature(data,threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     show_para=True):
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
    MDF = compute_MDF(data)
    MNF = compute_MNF(data)
    mDWT = compute_mDWT(data)
    feature = np.concatenate([IEMG,MAV,SSI,VAR,RMS,WL,ZC,SSC,WAMP,skew,Acti,AR,HIST,MDF,MNF,mDWT],axis =1)
    if show_para:
        print('threshold_WAMP:%0.1f, threshold_ZC:%0.1f, threshold_SSC:%0.1f,bins:%d,ranges:(%d,%d)'
          %(threshold_WAMP,threshold_ZC,threshold_SSC,bins,ranges[0],ranges[1]))
        print('IEMG,MAV,SSI,VAR,RMS,WL,ZC,SSC,WAMP,skew,Acti,AR,HIST,MDF,MNF,mDWT')
    return feature

def generate_feature_pd(data,threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     fbins=5,
                     franges=(0,300),
                     threshold_F=0.5,
                     num = 3,
                     wavelet='db7',
                     level=3):
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    IEMG = pd.DataFrame(compute_IEMG(data),columns=columns+'_IEMG')
    #MAV = pd.DataFrame(compute_MAV(data),columns=columns+'_MAV')
    SSI = pd.DataFrame(compute_SSI(data),columns=columns+'_SSI')
    #VAR = pd.DataFrame(compute_VAR(data),columns=columns+'_VAR')
    #RMS = pd.DataFrame(compute_RMS(data),columns=columns+'_RMS')
    WL = pd.DataFrame(compute_WL(data),columns=columns+'_WL')
    ZC = pd.DataFrame(compute_ZC(data,threshold_ZC),columns=columns+'_ZC')
    #ZC = compute_ZC_expand_pd(data,threshold_ZC)
    ku = pd.DataFrame(compute_ku(data),columns=columns+'_ku')
    SSC = pd.DataFrame(compute_SSC(data,threshold_SSC),columns=columns+'_SSC')
    WAMP = pd.DataFrame(compute_WAMP(data,threshold_WAMP),columns=columns+'_WAMP')
    skew = pd.DataFrame(compute_Skewness(data),columns=columns+'_skew')
    Acti = pd.DataFrame(compute_Acti(data),columns=columns+'_Acti')
    AR = pd.DataFrame(compute_AR(data),columns=columns+'_AR')
    #AR = compute_AR_pd(data)
    HIST = compute_HIST_pd(data,bins=bins,ranges=ranges)
    #FHIST = compute_FHIST_pd(data,bins=fbins,ranges=franges,threshold=threshold_F)
    MF = compute_MaxFreq_pd(data,num=num)
    MDF = pd.DataFrame(compute_MDF(data),columns=columns+'_MDF')
    MNF = pd.DataFrame(compute_MNF(data),columns=columns+'_MNF')
    mDWT = compute_mDWT_pd(data,wavelet,level)
    feature = pd.concat([IEMG,SSI,WL,ZC,ku,SSC,WAMP,skew,Acti,AR,HIST,MF,MDF,MNF,mDWT],axis =1)
    return feature

def pipeline_feature(path,width = 256,
                     columns = None,
                     stride = 32,
                     scaler=False,
                     threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     level=3,
                     show_para=True,
                     filt = None):
    emg_data = pd.read_csv(path)
    emg_data = emg_data.fillna({'LEFT_TA':emg_data.LEFT_TA.mean(),
                           'LEFT_TS':emg_data.LEFT_TS.mean(),
                           'LEFT_BF':emg_data.LEFT_BF.mean(),
                           'LEFT_RF':emg_data.LEFT_RF.mean(),
                           'RIGHT_TA':emg_data.RIGHT_TA.mean(),
                           'RIGHT_TS':emg_data.RIGHT_TS.mean(),
                           'RIGHT_BF':emg_data.RIGHT_BF.mean(),
                           'RIGHT_RF':emg_data.RIGHT_RF.mean()})
    if columns != None:
        emg_data = emg_data[['Time','Label1','Label2']+columns]
    x,y = generate_window_slide_data(emg_data,width=width,stride=stride,scaler=scaler)
    if filt != None:
        x = lowpass_filter(x,filt)
    feature = generate_feature(x,threshold_WAMP=threshold_WAMP,
                               threshold_ZC=threshold_ZC,
                               threshold_SSC=threshold_SSC,
                               bins=bins,
                               ranges=ranges,
                               show_para=show_para,
                               level=level)
    return feature,y

def pipeline_feature_pd(path,width = 256,
                     stride = 32,
                     scaler=False,
                     threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     fbins=5,
                     franges=(0,300),
                     threshold_F=0.5,
                     num = 3,
                     level=3,
                     filt = None,
                     drop_na=False,
                     same_label=False):
    emg_data = pd.read_csv(path)

    if drop_na:
        emg_data = emg_data.dropna().reset_index(drop=True)
        #print(emg_data)
        drop = False
    else:
        length = len(emg_data)
        na = emg_data.isna().sum()
        cri = na > length/10
        if any(cri):
            return True, []
        else:
            drop = False
        emg_data = emg_data.fillna({'LEFT_TA':emg_data.LEFT_TA.mean(),
                           'LEFT_TS':emg_data.LEFT_TS.mean(),
                           'LEFT_BF':emg_data.LEFT_BF.mean(),
                           'LEFT_RF':emg_data.LEFT_RF.mean(),
                           'RIGHT_TA':emg_data.RIGHT_TA.mean(),
                           'RIGHT_TS':emg_data.RIGHT_TS.mean(),
                           'RIGHT_BF':emg_data.RIGHT_BF.mean(),
                           'RIGHT_RF':emg_data.RIGHT_RF.mean()})
    if filt != None:
        fn = 10
        wn=2*fn/1000
        fn1 = 350
        wn1 = 2*fn1/1000
        #b, a = signal.butter(4, [wn,wn1], 'bandpass')
        b, a = signal.butter(1, wn, 'highpass')
        for i in ['LEFT_TA','LEFT_TS','LEFT_BF','LEFT_RF','RIGHT_TA','RIGHT_TS','RIGHT_BF','RIGHT_RF']:
            emg_data.loc[:,i] = signal.filtfilt(b, a, emg_data.loc[:,i])
    #emg_data.iloc[:,3:] = normalize(emg_data.iloc[:,3:])
    #mc = MinMaxScaler((-1,1))
    #emg_data.iloc[:,3:] = mc.fit_transform(emg_data.iloc[:,3:])
    x,y = generate_window_slide_data(emg_data,
                          width=width,
                          stride=stride,
                          scaler=scaler,
                          same_label=same_label)
    #if filt != None:
    #    x = bandpass_filter(x)
    Data = pd.DataFrame(y,columns=['Label'])
    feature = generate_feature_pd(x,threshold_WAMP=threshold_WAMP,
                               threshold_ZC=threshold_ZC,
                               threshold_SSC=threshold_SSC,
                               bins=bins,
                               ranges=ranges,
                               fbins=fbins,
                               franges=franges,
                               threshold_F=threshold_F,
                               level=level,
                               num = num)
    Data = Data.join(feature)
    Data['File']=path.split('/')[-1]
    return drop, Data

def pipeline_feature_pd2(path,width = 256,
                     stride = 32,
                     scaler=False,
                     threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     fbins=5,
                     franges=(0,300),
                     threshold_F=0.5,
                     num = 3,
                     level=3,
                     same_label=False):
    emg_data = pd.read_csv(path)

    emg_data = emg_data.dropna().reset_index(drop=True)

    fn = 20
    wn=2*fn/1000
    fn1 = 300
    wn1 = 2*fn1/1000
    b, a = signal.butter(4, [wn,wn1], 'bandpass')
    fs = 1000.0  # Sample frequency (Hz)
    f0 = 50  # Frequency to be removed from signal (Hz)
    Q = 100.0  # Quality factor
    b1, a1 = signal.iirnotch(f0, Q, fs)
    #b, a = signal.butter(4, wn, filt[1])
    for i in ['LEFT_TA','LEFT_TS','LEFT_BF','LEFT_RF','RIGHT_TA','RIGHT_TS','RIGHT_BF','RIGHT_RF']:
        emg_data.loc[:,i] = signal.filtfilt(b, a, emg_data.loc[:,i])
        emg_data.loc[:,i] = signal.filtfilt(b1, a1, emg_data.loc[:,i])
        ind = abs(zscore(emg_data.loc[:,i]))<10
        emg_data.loc[~ind,i]=emg_data.loc[ind,i].mean()
    sc = StandardScaler()
    emg_data.iloc[:,3:] = sc.fit_transform(emg_data.iloc[:,3:])
    
    x,y = generate_window_slide_data(emg_data,
                          width=width,
                          stride=stride,
                          scaler=scaler,
                          same_label=same_label)

    Data = pd.DataFrame(y,columns=['Label'])
    feature = generate_feature_pd(x,threshold_WAMP=threshold_WAMP,
                               threshold_ZC=threshold_ZC,
                               threshold_SSC=threshold_SSC,
                               bins=bins,
                               ranges=ranges,
                               fbins=fbins,
                               franges=franges,
                               threshold_F=threshold_F,
                               level=level,
                               num = num)
    Data = Data.join(feature)
    Data['File']=path.split('/')[-1]
    return Data

def pipeline_selected_feature(path,
                     columns = None,
                     drop_cols = None,
                     width = 256,
                     stride = 32,
                     scaler=False,
                     threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     show_para=True):
    emg_data = pd.read_csv(path)
    emg_data = emg_data.fillna({'LEFT_TA':emg_data.LEFT_TA.mean(),
                           'LEFT_TS':emg_data.LEFT_TS.mean(),
                           'LEFT_BF':emg_data.LEFT_BF.mean(),
                           'LEFT_RF':emg_data.LEFT_RF.mean(),
                           'RIGHT_TA':emg_data.RIGHT_TA.mean(),
                           'RIGHT_TS':emg_data.RIGHT_TS.mean(),
                           'RIGHT_BF':emg_data.RIGHT_BF.mean(),
                           'RIGHT_RF':emg_data.RIGHT_RF.mean()})
    if columns != None:
        emg_data = emg_data[['Time','Label1','Label2']+columns]
    if drop_cols != None:
        emg_data = emg_data.drop(columns = drop_cols)
    x,y = generate_window_slide_data(emg_data,width=width,stride=stride,scaler=scaler)
    feature = generate_feature(x,threshold_WAMP=threshold_WAMP,
                               threshold_ZC=threshold_ZC,
                               threshold_SSC=threshold_SSC,
                               bins=bins,
                               ranges=ranges,
                               show_para=show_para)
    return feature,y

def pipeline_cwt(path,
            width = 256,
            stride = 64,
            scaler = False,
            norm = False,
            width_c = 32,
            wavelet = 'mexh',
            same_label=False,
            dropna=True,
            filt=None):
    emg_data = pd.read_csv(path)
    if dropna:
        emg_data = emg_data.dropna().reset_index(drop=True)
    else:
        emg_data = emg_data.fillna({'LEFT_TA':emg_data.LEFT_TA.mean(),
                           'LEFT_TS':emg_data.LEFT_TS.mean(),
                           'LEFT_BF':emg_data.LEFT_BF.mean(),
                           'LEFT_RF':emg_data.LEFT_RF.mean(),
                           'RIGHT_TA':emg_data.RIGHT_TA.mean(),
                           'RIGHT_TS':emg_data.RIGHT_TS.mean(),
                           'RIGHT_BF':emg_data.RIGHT_BF.mean(),
                           'RIGHT_RF':emg_data.RIGHT_RF.mean()})
    
    emg_data.iloc[:,3:]=np.clip(emg_data.iloc[:,3:],-500,500)
    
    if filt != None:
        fn = filt
        wn=2*fn/1000
        fn1 = 350
        wn1 = 2*fn1/1000
        #b, a = signal.butter(4, [wn,wn1], 'bandpass')
        b, a = signal.butter(4, wn, 'lowpass')
        for i in ['LEFT_TA','LEFT_TS','LEFT_BF','LEFT_RF','RIGHT_TA','RIGHT_TS','RIGHT_BF','RIGHT_RF']:
            emg_data.loc[:,i] = signal.filtfilt(b, a, emg_data.loc[:,i])
            
    #ind = abs(zscore(emg_data.loc[:,i]))<10
    #emg_data.loc[~ind,i]=emg_data.loc[ind,i].mean()
            
    if norm:
        ms = MinMaxScaler()
        emg_data.iloc[:,3:] = ms.fit_transform(emg_data.iloc[:,3:])
        
    x,y = generate_window_slide_data(emg_data,
                          width=width,
                          stride=stride,
                          scaler=scaler,
                          same_label=same_label)
    cwt = generate_CWT_feature(x,scale=width_c,wavelet=wavelet)
    return cwt, y

def pipeline_dwt(path,
            width = 256,
            stride = 64,
            scaler = False,
            norm = False,
            level = 3,
            wavelet = 'db7',
            same_label=False):

    emg_data = pd.read_csv(path)
    emg_data = emg_data.dropna().reset_index(drop=True)
    if norm:
        emg_data.iloc[:,3:] = normalize(emg_data.iloc[:,3:])
    x,y = generate_window_slide_data(emg_data,
                          width=width,
                          stride=stride,
                          scaler=scaler,
                          same_label=same_label)
    dwt = compute_DWT(x,wavelet,level)
    
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    for m in range(len(dwt[0])//8):
        columns_b += ['_%d'%m]
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    feature = pd.DataFrame(dwt,columns=index)
    Data = pd.DataFrame(y,columns=['Label'])
    Data = Data.join(feature)
    Data['File']=path.split('/')[-1]
    return feature

def pipeline_cwt_feature(path,
                width = 256,
                stride = 64,
                scaler = False,
                norm = False,
                scale = 32,
                wavelet = 'mexh',
                filt = None,
                same_label=False):

    emg_data = pd.read_csv(path)
    emg_data = emg_data.dropna().reset_index(drop=True)
    if norm:
        emg_data.iloc[:,3:] = normalize(emg_data.iloc[:,3:])
    x,y = generate_window_slide_data(emg_data,
                          width=width,
                          stride=stride,
                          scaler=scaler,
                          same_label=same_label)
    if filt != None:
        x = lowpass_filter(x,filt)
    cwt = compute_CWT_feature(x,scale,wavelet)
    #print(cwt)
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = ['_Mean_Coe','_Min_Coe','_Mean_Scale','_Median_Scale']
    index = []
    columns_b = pd.Index(columns_b)
    for col in columns:
        index += (col+columns_b).to_list()
    feature = pd.DataFrame(cwt,columns=index)
    Data = pd.DataFrame(y,columns=['Label'])
    Data = Data.join(feature)
    Data['File']=path.split('/')[-1]
    return Data

def get_features_from_dwt(data,wavelet='db7',level=5):
    coes = pywt.wavedec(data,wavelet=wavelet,mode=0,level=level,axis=1)
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    feature = pd.DataFrame()
    for i in range(len(coes)):
        #IEMG = pd.DataFrame(compute_IEMG(coes[i]),columns=columns+'_IEMG')
        RMS = pd.DataFrame(compute_RMS(coes[i]),columns=columns+'_RMS%d'%i)
        #WL = pd.DataFrame(compute_WL(coes[i]),columns=columns+'_WL%d'%i)
        ZC = pd.DataFrame(compute_ZC(coes[i],1e-3),columns=columns+'_ZC%d'%i)
        ku = pd.DataFrame(compute_ku(coes[i]),columns=columns+'_ku%d'%i)
        #SSC = pd.DataFrame(dp.compute_SSC(coes[i],threshold_SSC),columns=columns+'_SSC%d'%i)
        WAMP = pd.DataFrame(compute_WAMP(coes[i],threshold_WAMP),columns=columns+'_WAMP%d'%i)
        skew = pd.DataFrame(compute_Skewness(coes[i]),columns=columns+'_skew%d'%i)
        Acti = pd.DataFrame(compute_Acti(coes[i]),columns=columns+'_Acti%d'%i)
        AR = pd.DataFrame(compute_AR(coes[i]),columns=columns+'_AR%d'%i)
        #AR = compute_AR_pd(coes[i])
        #HIST = compute_HIST_pd(coes[i],bins=bins,ranges=ranges)
        #FHIST = compute_FHIST_pd(coes[i],bins=fbins,ranges=franges,threshold=threshold_F)
        #MF = compute_MaxFreq_pd(coes[i],num=num)
        MDF = pd.DataFrame(dp.compute_MDF(coes[i]),columns=columns+'_MDF%d'%i)
        MNF = pd.DataFrame(dp.compute_MNF(coes[i]),columns=columns+'_MNF%d'%i)
        feature = pd.concat([feature,RMS,ZC,ku,WAMP,skew,Acti,AR,MDF,MNF],axis =1)
    return feature