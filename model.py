from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K
from tensorflow import keras
# from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.initializers import TruncatedNormal
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,confusion_matrix,accuracy_score
import tensorflow as tf
# from imblearn.over_sampling import SMOTE


import re
import datetime
import os
import numpy as np



def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)

import abc
class NNModel(metaclass=abc.ABCMeta):
    
    def __init__(self,name,mode,config,model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.name = name
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)
    
    @abc.abstractmethod
    def build(self,mode,config):
        pass
    
    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.name), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint
    
    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            keras_model = self.keras_model
            layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
                else keras_model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

        # Update the log directory
        self.set_log_dir(filepath)
    
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]"+self.name+"\_[\w-]+(\d{4})\.h5"
            # Use string for regex since we might want to use pathlib.Path as model_path
            m = re.match(regex, str(model_path))
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_{}_*epoch*.h5".format(self.name,
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        
    def train(self, train_dataset, val_dataset, epochs, batch_size, other_callbacks=None,class_weights=None,):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset.
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        binary: If True do the binary classification 0:others
        classes: classes for multi-classification
        """
        assert self.mode == "training", "Create model in training mode."

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Callbacks
        callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                histogram_freq=0, write_graph=True, write_images=False),
                 keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                   verbose=0, save_weights_only=True),]
        
        # Add other callbacks to the list
        if other_callbacks:
            callbacks += other_callbacks
            
        # Train
        log("\nStarting at epoch {}.\n".format(self.epoch))
        log("Checkpoint Path: {}".format(self.checkpoint_path))

        self.keras_model.fit(train_dataset[0],
                      train_dataset[1],
                      validation_data=(val_dataset[0],val_dataset[1]),
                      epochs=epochs,
                      batch_size=batch_size,
                      class_weight=class_weights,
                      callbacks=callbacks,
                      shuffle=True
        )
        self.epoch = max(self.epoch, epochs)
        
    def detect(self, data):
        """
        Runs the detection pipeline.
        """
        assert self.mode == "inference", "Create model in inference mode."

        # Run object detection
        results = self.keras_model.predict(data, verbose=0)

        return results
    
    def model_metrics(self,data,label):
        pred = self.keras_model.predict(data)
        acc = accuracy_score(np.argmax(label,axis=1),np.argmax(pred,axis=1))
        cm = confusion_matrix(np.argmax(label,axis=1),np.argmax(pred,axis=1))
        return acc,cm