
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    
    ### Parameters for data processing ###
    
    # Name the configurations. For example, 'XGBoost_data', 'Experiment 3', ...etc.
    NAME = None  # Override in sub-classes
    
    # The length of window for data segmentation
    WINDOW_SIZE = 1024
    
    # Step size for data segmentation
    STEP_SIZE = 512
    
    # Whether to scale the data
    SCALE = True
    
    # Whether to keep data with same label1 and label2
    SAME_LABEL = True
    
    # Lambda for detrend. None for not using detrend.
    DETREND_LAMBDA = 50
    
    # Cut-off frequency of lowpass filter. None for no using highpass filter
    FN_LP = 300
    
    # Channel to use
    CHANNELS = ['LEFT_TA','LEFT_TS','LEFT_BF','LEFT_RF',
                'RIGHT_TA','RIGHT_TS','RIGHT_BF','RIGHT_RF']
    
    # Files for test set
    TEST_FILES = ['G08_FoG_1_trial_1_emg.csv',
              'normal/G09_Walking_trial_2_emg.csv',
              'normal/G09_Walking_trial_4_emg.csv',
              'normal/G09_Walking_trial_6_emg.csv',
              'normal/G11_Walking_trial_2_emg.csv',
              'normal/G11_Walking_trial_4_emg.csv',
              'normal/P231_M050_A_Walking_trial_2_emg.csv']
    
    # Ratio of training set
    TRAIN_SET_RATIO = 0.8
    
    # Whether to shuffle the data
    SHUFFLE = True
    
    # Whether to rectify the signal
    RECT = False
    
    # The number of neighbors for envelope calculating
    N_ENV = 20

    ### Parameter for training ###
    
    # Number of classification classes
    NUM_CLASSES = 2 # override in subclasses
    
    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        # for a in dir(self):
        #     if not a.startswith("__") and not callable(getattr(self, a)):
        #         print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
    
class Config_f(Config):
    """Base configuration class for feature extraction. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Features ot use
    FEATURES_LIST = ['IEMG','SSI','WL','ZC','ku','SSC','skew',
                     'Acti','AR','HIST','MDF','MNF','mDWT']
    
    # Threshold for willison amplitude
    THRESHOLD_WAMP = 1
    
    # Threshold for zero crossing
    THRESHOLD_ZC = 0.
    
    # Threshold for slope sign crossing
    THRESHOLD_SSC = 0.01
    
    # The number of bins for EMG Histogram
    BINS = 3
    
    # The lower and upper range of the bins for EMG Histogram
    RANGES = (-3,3)
    
    # The number of bins for Frequency Histogram
    FBINS = 5
    
    # The lower and upper range of the bins for Frequency Histogram
    FRANGES = (0,300)
    
    # The minimum frequency amplitude which is used for Frequency Histogram
    THRESHOLD_F = 0.5
    
    # The number how much of the largest frequency is used for MaxFreq
    NUM_MF = 3
    
    # Wavelet of discrete wavelet transform
    WAVELET_DWT = 'db7'
    
    # Decomposition level (must be >= 0). If level is None (default) then it will be calculated using the dwt_max_level function.
    LEVEL_DWT = 3