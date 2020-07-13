from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import numpy as np

def scale_data(data,scaler):
    X = data.iloc[:,3:]
    X = sc.fit_transform(X)
    data.iloc[:,3:] = X
    return data

def generate_window_slide_data(data,width = 256,stride = 32):
    l = len(data)
    end = (l-width)//stride+1
    X = []
    Y = []
    sc = StandardScaler()
    for i in range(end):
        if len(set(data.Label2[i*stride:i*stride+width])) == 1:
                Y += [data.Label2[i*stride]]
                #x_sc = sc.fit_transform(np.array(data.iloc[i*stride:i*stride+width,3:]))
                #X += [x_sc]
                X += [np.array(data.iloc[i*stride:i*stride+width,3:])]
            #print(set(data.Label2[i*stride:i*stride+width]))
        else:
            #print(set(data.Label2[i*stride:i*stride+width]))
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

def generate_feature(data):
    IEMG = compute_IEMG(data)
    MAV = compute_MAV(data)
    SSI = compute_SSI(data)
    VAR = compute_VAR(data)
    RMS = compute_RMS(data)
    WL = compute_WL(data)
    ZC = compute_ZC(data)
    SSC = compute_SSC(data)
    WAMP = compute_WAMP(data,50)
    skew = compute_Skewness(data)
    Acti = compute_Acti(data)
    feature = np.concatenate([IEMG,MAV,SSI,VAR,RMS,WL,ZC,SSC,WAMP,skew,Acti],axis =1)
    return feature