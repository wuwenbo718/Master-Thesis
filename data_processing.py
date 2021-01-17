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

def lowpass_filter(data):
    """
    Apply lowpass filter on n-dimension data
    """
    x = np.zeros(data.shape)
    N,M = data.shape[0::2]
    wn=2*fn/1000
    b, a = signal.butter(4, wn, 'lowpass')
    for i in range(N):
        for j in range(M):
            x[i,:,j] = signal.filtfilt(b, a, data[i,:,j])
    return x

def bandpass_filter(data,fn=350):
    """
    Apply bandpass filter on n-dimension data
    """
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

def generate_window_slide_data(data,width = 256,stride = 64,scaler=False,same_label=False):
    """
    Segment the signal.
    
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
                X += [x_sc]
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


def generate_window_slide_data_time_continue_fremove(data,width = 256,stride = 64,scaler=False,same_label=False):
    """
    Segment the signal. Only keep the segments with continuous time and low frequency
    components that not much bigger than the other components.
    
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
                skip = False
                for j in range(temp.shape[-1]):
                    freqs, power=signal.periodogram(temp[:,j], 1e3)
                    ind_l = freqs<20
                    max_l = np.max(power[ind_l])
                    max_h = np.max(power[~ind_l])
                    # The amplitude of frequency components which are lower than 20 Hz must be less than 10 times of the other components and the amplitude of max frequency must over 0.5
                    if (max_l>10*max_h) | (max_h<0.5):
                        skip = True
                        break
                if skip:
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
                skip = False
                for j in range(temp.shape[-1]):
                    freqs, power=signal.periodogram(temp[:,j], 1e3)
                    ind_l = freqs<20
                    max_l = np.max(power[ind_l])
                    max_h = np.max(power[~ind_l])
                    if (max_l>10*max_h) | (max_h<0.5):
                        skip = True
                        break
                if skip:
                    continue
                Y += [data.Label2[i*stride]]
                X += [temp]
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)

def generate_window_slide_data_time_continue(data,width = 256,stride = 64,scaler=False,same_label=False):
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
                Y += [data.Label2[i*stride]]
                X += [temp]
            else:
                continue
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.uint8)



def generate_CWT_feature(data,scale=32,wavelet = 'mexh'):
    """
    Compute continuous wavelet transform of EMG signal
    """
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
    """
    Compute features based on continuous wavelet transform of EMG signal
    """
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
    """
    Compute discrete wavelet transform of EMG signal
    """
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

def compute_CC_pd(data,p=4):
    
    '''
    Compute Cepstral Coefficient of data matrix (pandas output version)
    
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
    feature = np.zeros((N,p*C))
    
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    
    for m in range(p):
        columns_b += ['_CC%d'%m]
    columns_b = pd.Index(columns_b)
    
    for col in columns:
        index += (col+columns_b).to_list()
    
    for i in range(N):
        for j in range(C):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,j*p:(j+1)*p] = compute_cc(ak,p)
            
    return pd.DataFrame(feature,columns=index)

def compute_AR_pd(data,p=4):
    """
    Compute Autoregression Coefficient (pandas output version)
    
    Inputs:
    data: EMG signal
    p: Model order
    
    Outputs:
    Autoregression Coefficient
    
    """
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

def compute_HIST_pd(data,bins=9,ranges=(-10,10)):
    """
    Compute EMG Histogram (pandas output version)
    
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
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    columns_b = []
    index = []
    
    for m in range(bins):
        columns_b += ['_HIST%d'%m]
    columns_b = pd.Index(columns_b)
    
    for col in columns:
        index += (col+columns_b).to_list()
    
    for i in range(N):
        for j in range(M):
            hist,_ = np.histogram(data[i,:,j],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
            
    return pd.DataFrame(feature,columns=index)

def compute_FHIST_pd(data,bins=5,ranges=(0,300),threshold = 0.5):
    """
    Compute EMG frequency Histogram
    
    Inputs:
    data: EMG signal
    bins: the number of bins
    ranges: the lower and upper range of the bins
    threshold: the minimum frequency amplitude which is used
    
    Outputs:
    Histogram
    """
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
        
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            ind = (power > threshold)
            hist,_ = np.histogram(freqs[ind],bins=bins,range=ranges)
            feature[i,j*bins:(j+1)*bins] = hist
            
    return pd.DataFrame(feature,columns=index)    

def compute_MaxFreq_pd(data,num=3):
    """
    Compute max frequency of EMG signal (pandas output version)
    
    Inputs:
    data: EMG signal
    num: The number how much of the largest frequency is used
    
    Outputs:
    Max frequency
    """
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
    
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            ind = np.argsort(-power)[:num]
            feature[i,j*num:(j+1)*num] = freqs[ind]
            
    return pd.DataFrame(feature,columns=index)  

def compute_MDF(data):
    """
    Compute median frequency 
    """
    N,M = data.shape[0::2]
    feature = np.zeros((N,M))
    
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            total = (cumtrapz(power,freqs))
            w=np.where(total>=(total[-1]/2))[0][0]
            feature[i,j] = freqs[w]
            
    return feature

def compute_MNF(data):
    """
    Compute mean frequency
    """
    N,M = data.shape[0::2]
    feature = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            freqs, power=signal.periodogram(data[i,:,j], 1e3)
            feature[i,j] = freqs@power/power.sum()
    return feature

def mDWT(data,wavelet='db7',level=3):
    """
    For compute_mDWT
    """
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
    """
    Compute Marginal Discrete Wavelet Transform
    """
    N,M = data.shape[0::2]
    feature = []
    for i in range(N):
        temp = []
        for j in range(M):
            temp.extend(mDWT(data[i,:,j],wavelet,level))
        feature.append(temp)
    return feature

def compute_mDWT_pd(data,wavelet='db7',level=3):
    """
    Compute Marginal Discrete Wavelet Transform (pandas output version)
    """
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
        print('threshold_WAMP:%0.1f, threshold_ZC:%0.1f, threshold_SSC:%0.1f,bins:%d,ranges:(%d,%d)'
          %(threshold_WAMP,threshold_ZC,threshold_SSC,bins,ranges[0],ranges[1]))
        print(feature_list)
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
    """
    Generate features (pandas output version)
    """
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
    Mobi = pd.DataFrame(compute_Mobi(data),columns=columns+'_Mobi')
    Comp = pd.DataFrame(compute_complexity(data),columns=columns+'_Comp')
    AR = pd.DataFrame(compute_AR(data),columns=columns+'_AR')
    #AR = compute_AR_pd(data)
    CC = compute_CC_pd(data)
    HIST = compute_HIST_pd(data,bins=bins,ranges=ranges)
    #FHIST = compute_FHIST_pd(data,bins=fbins,ranges=franges,threshold=threshold_F)
    MF = compute_MaxFreq_pd(data,num=num)
    MDF = pd.DataFrame(compute_MDF(data),columns=columns+'_MDF')
    MNF = pd.DataFrame(compute_MNF(data),columns=columns+'_MNF')
    mDWT = compute_mDWT_pd(data,wavelet,level)
    feature = pd.concat([IEMG,SSI,WL,ZC,ku,SSC,WAMP,skew,Acti,Mobi,Comp,AR,CC,HIST,MF,MDF,MNF,mDWT],axis =1)
    return feature




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

def get_dwt(data,wavelet='db7',level=5,mode=0):
    
    coes = pywt.wavedec(data,wavelet=wavelet,mode=mode,level=level,axis=1)
    n,l,c = coes[-1].shape
    feature = np.zeros((n,l,c,0))
    
    for i in range(len(coes)-1):
        temp = signal.resample(coes[i],l,axis=1)[:,:,:,np.newaxis]
        feature = np.concatenate([feature,temp],axis=3)

    feature = np.concatenate([feature,coes[-1][:,:,:,np.newaxis]],axis=3)
    
    return feature