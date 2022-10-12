__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

import os
import wandb
from tensorflow import keras
# from keras.api._v2 import keras


import FIDN

# Turn on mixed precision training
# train the model with float16 to speed up and reduce memory usage
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' # 

# Load Dataset
print('Start Loading Dataset...')
fpath = '/Users/Bloomberg/Downloads/all_304_fire_combine_all_feature_512_wind_precipitation_new.npy'

dataset, train_dataset, val_dataset, test_dataset, \
x_train, y_train, x_val, y_val, x_test, y_test = FIDN.dataset.setup_dataset(
    fpath)
print('Dataset Loaded Successful!')

# Global configuration of training
global_config = dict(
    version='1.0',
    save_path='./models',
)

def train_fidn_model():
    """
    Build FIDN model and perform training

    Returns:
        A trained keras.Model instance of FIDN model
    """
    # Local configuration of training
    config = dict(
        epochs=100,
        batch_size=10,
        name='fidn',
        optimizer='adam',
        loss='binary_crossentropy',
    )
    config = dict(config, **global_config)

    print('Start Training Combine Model, Project config:', config)
    # Monitoring training losses and metrics with wandb
    with wandb.init(config=config,
                    project="forcast_wildfire_fidn",
                    entity="irp-bp221"):
        config = wandb.config
        # Construct input layer
        inp = keras.layers.Input(shape=(x_train.shape[1:]))
        # Build FIDN model
        fidn_model = FIDN.models.fidn.build_fidn_model(inp)
        fidn_model.compile(
            optimizer=config.optimizer,
            loss=config.loss,
            metrics=[FIDN.losses.ssim_metrics,
                     FIDN.losses.psnr_metrics,
                     FIDN.losses.custom_mean_squared_error,
                     FIDN.losses.relative_root_mean_squared_error]
        )
        
        # Visualize the model structure
        print(fidn_model.summary())
        keras.utils.plot_model(
            fidn_model, to_file=f'{config.version}_{config.name}.png',
            show_shapes=True)

        # Define wandb callback to monitor training.
        wandb_callback = wandb.keras.WandbCallback(save_model=False)
        # Fit the model to the training data.
        fidn_model.fit(
            x_train,
            y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(x_val, y_val),
            callbacks=[wandb_callback]
        )
        # Save model to disk
        os.makedirs(f'{config.save_path}', exist_ok=True)
        save_path = f'{config.save_path}/fidn_{config.version}_{config.name}_' + \
                    f'epoch{config.epochs}_batchsize{config.batch_size}.h5'
        fidn_model.save(save_path)
        print('Model saved at ', save_path)
        return fidn_model


if __name__ == '__main__':
    train_fidn_model()
