__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from tensorflow import keras
from keras.api._v2 import keras
import FIDN.losses as losses
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
This file stores the auxiliary functions associated 
with the evaluation of the model
"""


def visual_origin_data(dataset, index):
    """
    Plot each layer of the dataset onto a single image

    Args:
        dataset: Raw data set to be read
        index: Index of fire events in the dataset

    Returns:
        A figure with 15 subfigures
    """
    data_channels = {
        0: 'burned area day 0',
        1: 'burned area day 1',
        2: 'burned area day 2',
        3: 'biomass above ground',
        4: 'biomass under ground',
        5: 'land slope',
        6: 'tree density',
        7: 'grass density',
        8: 'bare density',
        9: 'snow',
        10: 'water',
        11: 'wind direction u',
        12: 'wind direction v',
        13: 'rain + snow',
        14: 'final burned area'
    }

    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))

    # Plot each of the sequential images for one random data example.
    for idx, ax in enumerate(axes.flat):
        img = np.squeeze(dataset[index][..., idx])
        if 2 < idx < 14:
            border_size = int((img.shape[0] - 128) / 2)
            img = img[border_size:-border_size, border_size:-border_size]
        im = ax.imshow(img, cmap="viridis")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='7%', pad='2%')
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title(f"{data_channels[idx]}")
        # ax.axis("off")

    # Print information and display the figure.
    plt.tight_layout()
    # plt.show()


def evaluate_and_save_pic(model, config, dataset):
    """
    Use the input model to make predictions on the input
    dataset and store pictures of the prediction results.
    The model output is compared with the true values,
    and metrics such as MSE, RRMSE, SSIM and PSNR are
    calculated and stored in a csv file.
    
    Args:
        model: Model to be evaluated
        config (dict): Output-related configuration
        dataset: Data set used for the evaluation

    Returns:
        A pandas DataFrame that stores the model's
        performance metrics on each sample
    """
    result_path = f'./result/result_{config["version"]}/{config["name"]}'
    os.makedirs(result_path, exist_ok=True)
    print('Start Evaluate Model, Save path: ', result_path)

    metrics = {
        "index": [],
        "ssim": [],
        "psnr": [],
        "mse": [],
        "rrmse": []
    }
    for index, example in enumerate(dataset):
        # Visual Origin Data
        visual_origin_data(dataset, index)
        plt.savefig(result_path + f'/{index}_info.jpg')
        plt.close()
        sys.stdout.write(f'\rProcessing Fire {index + 1} / {len(dataset)}')
        # Pick the first 14 channel from the input
        frames = example[..., :-1]
        original_frames = example[..., -1:]
        original_frames = original_frames.astype(np.float32)

        # Extract the model's prediction and post-process it.
        predicted_frame = model.predict(
            np.expand_dims(frames, axis=0), verbose=0)
        # Binary data
        # predicted_frame[predicted_frame >= config['threshold']] = 1
        # predicted_frame[predicted_frame < config['threshold']] = 0
        metrics['index'].append(index)
        metrics['ssim'].append(losses.ssim_metrics(
            np.expand_dims(original_frames, axis=0),
            np.expand_dims(predicted_frame, axis=0)
        ).numpy())
        metrics['psnr'].append(losses.psnr_metrics(
            np.expand_dims(original_frames, axis=0),
            np.expand_dims(predicted_frame, axis=0)
        ).numpy())
        metrics['mse'].append(np.mean(keras.metrics.mean_squared_error(
            np.expand_dims(original_frames, axis=0),
            np.expand_dims(predicted_frame, axis=0)
        ).numpy()))
        metrics['rrmse'].append(
            np.mean(losses.relative_root_mean_squared_error(
                np.expand_dims(original_frames, axis=0),
                np.expand_dims(predicted_frame, axis=0)
            ).numpy()))
        # Construct a figure for the original and new frames.
        _, axes = plt.subplots(1, 3, figsize=(8, 4))
        # Plot the original frames.
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
        ax1.imshow(np.squeeze(frames[..., 2]), cmap="viridis")
        ax1.set_title(f"Burned area day 2")
        ax1.axis("off")
        ax2.imshow(np.squeeze(original_frames), cmap="viridis")
        ax2.set_title(f"Final burned area")
        ax2.axis("off")
        ax3.imshow(np.squeeze(predicted_frame), cmap="viridis")
        ax3.set_title(f"Predict")
        ax3.axis("off")
        # Display the figure.
        plt.savefig(result_path + f'/{index}.jpg')
        plt.close()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index('index', inplace=True)
    metrics_df.to_csv(result_path + f'/metrics.csv')
    return metrics_df


def load_model(model_path):
    """
    Reading model from files
    
    Args:
        model_path: Storage paths for models

    Returns:
        A keras model
    """
    fidn_model = keras.models.load_model(model_path, custom_objects={
        'ssim_metrics': losses.ssim_metrics,
        'psnr_metrics': losses.psnr_metrics,
        'custom_mean_squared_error': losses.custom_mean_squared_error,
        'relative_root_mean_squared_error': losses.relative_root_mean_squared_error,
    })
    return fidn_model
