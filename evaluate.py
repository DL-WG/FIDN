__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

import os
import tensorflow as tf


import FIDN

# Turn on mixed precision training
# train the model with float16 to speed up and reduce memory usage
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

print('Start Loading Dataset...')
fpath = './all_304_fire_combine_all_feature_512_wind_precipitation_new.npy'

# Load Dataset
dataset = FIDN.dataset.load_dataset(fpath)
train_dataset, val_dataset, test_dataset = FIDN.dataset.split_dataset(dataset)
print('Dataset Loaded Successful!')

if __name__ == '__main__':
    config = dict(
        version='1.0',
        model_path='./models/fidn_1.0_fidn_epoch100_batchsize16.h5',
        name='fidn',
    )
    # Load Model in config
    fidn_model = FIDN.evaluate.load_model(config['model_path'])
    # Evaluate the model and 
    # save the result pic and metrics csv in ./result/
    FIDN.evaluate.evaluate_and_save_pic(fidn_model, config, test_dataset)
