from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import numpy as np
import pandas as pd
from nitime.algorithms.autoregressive import AR_est_YW
import pywt
from scipy import signal
from scipy.integrate import cumtrapz

def scale_data(data,scaler):
    X = data.iloc[:,3:]
    X = sc.fit_transform(X)
    data.iloc[:,3:] = X
    return data

def lowpass_filter(data,fn=250):
    x = np.zeros(data.shape)
    N,M = data.shape[0::2]
    wn=2*fn/1000
    b, a = signal.butter(8, wn, 'lowpass')
    for i in range(N):
        for j in range(M):
            x[i,:,j] = signal.filtfilt(b, a, data[i,:,j])
    return x

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


def generate_CWT_feature(data,scale=32,wavelet = 'mexh'):
    n,t,c = data.shape
    cwtmatr = np.zeros((n,scale,t,c))
    for i in range(n):
        for j in range(c):
            cwtmatr[i,:,:,j],_ = pywt.cwt(data[i,:,j],np.arange(1,scale+1),wavelet)
    return cwtmatr

def compute_CWT_feature(data,scale=32,wavelet = 'mexh'):
    n,t,c = data.shape
    cwt = np.zeros((n,4*c))
    #print(cwt.shape)
    scales = np.arange(1,scale+1)
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
    M = data.shape[-1]
    feature = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ak,_ = AR_est_YW(data[i,:,j],p)
            feature[i,j] = ak[0]
    return feature

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

def mDWT(data):
    wa = pywt.wavedec(data,'db7',3)
    wa = np.concatenate(wa)
    N = len(wa)
    S = int(np.log2(N))
    M = []
    for i in range(S):
        C = N//(2**(i+1))-1
        #print(C)
        M.append(np.abs(wa[:C+1]).sum())
    return M
    
def compute_mDWT(data):
    N,M = data.shape[0::2]
    feature = []
    for i in range(N):
        temp = []
        for j in range(M):
            temp.extend(mDWT(data[i,:,j]))
        feature.append(temp)
    return feature

def compute_mDWT_pd(data):
    N,M = data.shape[0::2]
    feature = []
    for i in range(N):
        temp = []
        for j in range(M):
            temp.extend(mDWT(data[i,:,j]))
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
                     ranges=(-10,10)):
    columns = pd.Index(['LEFT_TA', 'LEFT_TS', 'LEFT_BF', 'LEFT_RF',
       'RIGHT_TA', 'RIGHT_TS', 'RIGHT_BF', 'RIGHT_RF'])
    IEMG = pd.DataFrame(compute_IEMG(data),columns=columns+'_IEMG')
    #MAV = pd.DataFrame(compute_MAV(data),columns=columns+'_MAV')
    SSI = pd.DataFrame(compute_SSI(data),columns=columns+'_SSI')
    #VAR = pd.DataFrame(compute_VAR(data),columns=columns+'_VAR')
    #RMS = pd.DataFrame(compute_RMS(data),columns=columns+'_RMS')
    WL = pd.DataFrame(compute_WL(data),columns=columns+'_WL')
    ZC = pd.DataFrame(compute_ZC(data,threshold_ZC),columns=columns+'_ZC')
    SSC = pd.DataFrame(compute_SSC(data,threshold_SSC),columns=columns+'_SSC')
    WAMP = pd.DataFrame(compute_WAMP(data,threshold_WAMP),columns=columns+'_WAMP')
    skew = pd.DataFrame(compute_Skewness(data),columns=columns+'_skew')
    Acti = pd.DataFrame(compute_Acti(data),columns=columns+'_Acti')
    AR = pd.DataFrame(compute_AR(data),columns=columns+'_AR')
    HIST = compute_HIST_pd(data,bins=bins,ranges=ranges)
    MDF = pd.DataFrame(compute_MDF(data),columns=columns+'_MDF')
    MNF = pd.DataFrame(compute_MNF(data),columns=columns+'_MNF')
    mDWT = compute_mDWT_pd(data)
    feature = pd.concat([IEMG,SSI,WL,ZC,SSC,WAMP,skew,Acti,AR,HIST,MDF,MNF,mDWT],axis =1)
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
                               show_para=show_para)
    return feature,y

def pipeline_feature_pd(path,width = 256,
                     stride = 32,
                     scaler=False,
                     threshold_WAMP=30,
                     threshold_ZC=0,
                     threshold_SSC=0,
                     bins=9,
                     ranges=(-10,10),
                     filt = None,
                     drop_na=False):
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
    x,y = generate_window_slide_data(emg_data,width=width,stride=stride,scaler=scaler)
    if filt != None:
        x = lowpass_filter(x,filt)
    Data = pd.DataFrame(y,columns=['Label'])
    feature = generate_feature_pd(x,threshold_WAMP=threshold_WAMP,
                               threshold_ZC=threshold_ZC,
                               threshold_SSC=threshold_SSC,
                               bins=bins,
                               ranges=ranges)
    Data = Data.join(feature)
    Data['File']=path.split('/')[-1]
    return drop, Data

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
            width_c = 32,
            wavelet = 'mexh'):
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
    cwt = generate_CWT_feature(x,widths=width_c,wavelet=wavelet)
    return cwt, y

def pipeline_dwt(path,
            width = 256,
            stride = 64,
            scaler = False,
            level = 3,
            wavelet = 'db7'):

    emg_data = pd.read_csv(path)
    emg_data = emg_data.dropna().reset_index(drop=True)
    x,y = generate_window_slide_data(emg_data,width=width,stride=stride,scaler=scaler)
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
            scale = 32,
            wavelet = 'mexh'):

    emg_data = pd.read_csv(path)
    emg_data = emg_data.dropna().reset_index(drop=True)
    x,y = generate_window_slide_data(emg_data,width=width,stride=stride,scaler=scaler)
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