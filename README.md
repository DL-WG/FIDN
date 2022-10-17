# FIDN

Prediction of wildfire final burned area with image-based Machine learning

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-important.svg)](https://github.com/ese-msc-2021/irp-bp221/blob/main/LICENSE) ![version](https://img.shields.io/badge/version-1.0-brightgreen)![python](https://img.shields.io/badge/python-v3.9.13-yellow)![python](https://img.shields.io/badge/tensorflow-v2.9.1-orange)

![plarform](https://img.shields.io/badge/platform-linux--64%20%7C%20win--32%20%7C%20osx--64%20%7C%20win--64-lightgrey)![build](https://img.shields.io/badge/build-passing-brightgreen)

## Introduction

Predicting the final burn area of a wildfire is essential to reduce economic losses and environmental damage caused by wildfires. However, this is an extremely challenging task due to the complexity and diversity of factors which influence wildfires. Existing works and models need further improvement in terms of computational time, computational cost and forecast accuracy. In this paper, we introduce image-based machine learning into this field. Based on the advanced Densely Connected Convolutional Network (DenseNet), we propose a new wildfire prediction model, Fire-Image-DenseNet (FIDN). This model integrates geographic and meteorological parameters such as biomass, land slope, tree, grass density, wind, snow, water and precipitation. We then applied the model to wildfires in the western US mainly located in California in recent years. After comparison with satellite imagery, the model can forecast the final burn area of the wildfires accurately and quickly.

In summary, we have made following contributions in this work:

- A predictive model, Fire-Image-DenseNet (FIDN), for predicting the final burned area of wildfire based on remote sensing and climate data is proposed. The model accepts as input images of current wildfire-burned areas preprocessed from Moderate Resolution Imaging Spectroradiometer(MODIS) satellite observations based on the latitude and longitude coordinates of the area and data of relevant geographical and meteorological features (e.g. vegetation, water, precipitation) extracted from the Oak Ridge National Laboratory (ORNL) DACC data repository, the Project for On-Board Autonomy - Végétation (PROBA-V) satellite, etc.
- The proposed model benefits from DenseNet, which, with taking full account of multi-dimensional parameters, it still significantly reduces computational costs compared to traditional forecasting models.
- The proposed model yields very promising and interpretable results, which reduces the average computation time by 99.92%, improves the structural similarity (SSIM) by 6%, improves the peak signal-to-noise ratio (PSNR) by 23% and reduces the mean square error (MSE) by 82% compared to the state-of-the-art CA model.
- The proposed model is general which does not require separate adjustment of parameters for different fires, as is the case with surrogate model and some physical models.

![model_structure](picture/model_structure.jpg)

## Project Structure

```
irp-bp221
├── CA
│   ├── CA_firebis.py - Scripts for running CA models
│   └── data.zip - Manually extracted data for use in CA models
├── FIDN - FIDN module
│   ├── init.py
│   ├── dataset.py - Includes code for dataset reading and pre-processing
│   ├── evaluate.py - Includes helper functions related to model evaluation
│   ├── losses.py - Includes the loss used for training and other reference metrics
│   └── models
│       ├── init.py
│       ├── densenet.py - Basic structure of FIDN based on DenseNet
│       └── fidn.py - Module for constructing the FIDN model
├── LICENSE
├── README.md
├── benchmark.ipynb - Quantitative comparison of CA and FIDN models
├── docs
│   └── FIDN Documentation.pdf - Documentation for the FIDN module
├── evaluate.py - Validate saved models, store images and performance metrics
├── info
│   ├── README.md
│   └── info.json
├── picture
│   └── model_structure.jpg
├── reports
│   ├── README.md
│   ├── bp221-final-report.pdf
│   └── bp221-project-plan.pdf
└── train.py - Constructing and training a FIDN model 
```

## Code Metadata

The project was developed and debugged on a private high performance computing server. The device is equipped with an Nvidia A40 graphics card with 48GB of video memory, Intel(R) Xeon(R) Gold 6330 processor with 28 cores and 56 threads, and 80GB of RAM, running Ubuntu 20.04 Operating System

This project was developed entirely in Python version 3.9.13. The main part of the project is based on the Tensorflow 2.9.1 development framework and the Keras backend, which was used to build and train the model. The training process uses a third-party library, wandb, developed by Weight\&Biases, to track changes in loss and metrics and changes in model parameters for each epoch of the training and validation sets throughout the project. To ensure the best performance of the models, we installed a numpy distribution based on the Intel MKL backend, which has a faster computational performance than the OpenBLAS backend on CPUs sold by Intel Corporation. For the data mentioned in the previous section, we used the Google Earth Engine (GEE) Python API to obtain it and uses Geopandas, Genjson, FINOA, PyProj and other third-party libraries to read and process geospatial data and longitude and latitude coordinate data.

The project resulted in the delivery of a Python package called ***FIDN***, which contains several modules covering dataset processing, Loss functions and Metrics, model construction and validation, and is fully annotated, which can be accessed in the Github repository.

## Dependencies

The third party libraries used in this project are as follows:

```
keras==2.9.0
matplotlib==3.2.2
numpy==1.23.1
pandas==1.4.3
Pillow==9.2.0
pyproj==3.3.0
pytest==7.1.2
Shapely==1.7.1
tensorflow==2.9.1
wandb==0.12.21
```

You can install all dependencies directly using the following command

```bash
pip install -r requirements.txt
pip3 install -r requirements.txt
```

## Access to datasets

You can download the processed dataset used in this study from this [link](https://drive.google.com/file/d/1pRUVeH7CiGfo_YgkXaA_PkEXP4yNQvRM/view?usp=sharing), which contains a total of 304 fire events in the western United States (mainly California 2012 to 2019) for the year, each fire event consisting of 15 images stored in a different channel. The meaning of each channel can be found in the table below, or you can click on the link to download to the raw data.

| Channel No. | Description                                             | Source     | Resolution   | Link                                                                                                             |
| ----------- | ------------------------------------------------------- | ---------- | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| 1           | burn area in day 0                                      | MODIS      | about 1km    | [Download](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1642)                                                    |
| 2           | burn area in day 1                                      | MODIS      | about 1km    | [Download](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1642)                                                    |
| 3           | burn area in Day 2                                      | MODIS      | about 1km    | [Download](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1642)                                                    |
| 4           | biomass above ground                                    | ORNL  DACC | 300 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_biomass_carbon_density_v1)         |
| 5           | biomass below ground                                    | ORNL  DACC | 300 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_biomass_carbon_density_v1)         |
| 6           | slope                                                   | CSP        | 270 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_mTPI)               |
| 7           | tree density                                            | PROBA-V    | 100 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) |
| 8           | grass density                                           | PROBA-V    | 100 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) |
| 9           | bare density                                            | PROBA-V    | 100 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) |
| 10          | snow density                                            | PROBA-V    | 100 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) |
| 11          | water density                                           | PROBA-V    | 100 meters   | [Download](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) |
| 12          | 10m u-component of wind(monthly average)                | ERA5       | 27830 meters | [Download](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY)                          |
| 13          | 10m v-component of wind(monthly average)                | ERA5       | 27830 meters | [Download](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY)                          |
| 14          | total precipitation(rainfall + snowfall) (monthly sums) | ERA5       | 27830 meters | [Download](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY)                          |
| 15          | final burn area                                         | MODIS      | about 1km    | [Download](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1642)                                                    |

## Usage

### FIDN Module

The model for this project can be easily integrated into your existing work environment, you just need to install the dependency package in `requirements.txt`, copy the `FIDN` folder to your workspace and import it using the following command The detailed documentation for this module is stored in `docs/FIDN Documentation.pdf`.

```python
import FIDN

FIDN.models.
FIDN.losses.
FIDN.evaluate.
FIDN.dataset.
```

### CA Module

The CA module script is tied to the manually extracted data. To use the script you first need to unzip `CA/data.zip` into the `CA/data` folder and then execute it using the following command. The `data.zip`includes 30 sets of wildfire events in the western United States, including information on vegetation type, vegetation density, slope, wind speed, wind direction, latitude and longitude, and more. The data includes images of wildfires burning on day 0, day 1 and day 2, and you can control which day the simulation starts from by using the `start_from` variable in the code. This information was extracted from the [IFTDSS](https://iftdss.firenet.gov) and the raw data can be accessed from this [link](https://drive.google.com/file/d/1v0ac1aD2ko1or0CWtlqAGQJUSIKZpVik/view?usp=sharing).

```bash
cd CA/
python CA_firebis.py -i <index of fire in test set>
```

### Train and Evaluate Scripts

The root directory of this project contains two script files, `train.py` and `evaluate.py`, which call the functions of the FIDN module and implement the model training and validation functions.

#### train.py

There is a uniform configuration area for the parameters associated with model training.

##### Global Config

Here you can configure the current version number of the model and the location where the model is stored. The version number of the model will appear in the output model name to distinguish the version.

```python
global_config = dict(
    version='1.0',
    save_path='./models',
)
```

##### Local Config

Here you can configure the epochs, batch size, loss, optimizer and give the current model an alias.

```python
config = dict(
    epochs=100,
    batch_size=16,
    name='fidn',
    optimizer='adam',
    loss='binary_crossentropy',
)
```

Once the configuration is complete use the following command to execute:

```python
python train.py
```

#### evaluate.py

This script also contains a configuration section where you can specify the model path, version and name (this will affect the path to save)

```python
config = dict(
    version='1.0',
    model_path='./models/fidn_1.0_fidn_epoch100_batchsize16.h5',
    name='fidn',
)
```

Once the configuration is complete use the following command to execute:

```
python evaluate.py
```

The output images and indicators will be stored in `./result` directory

### Benchmark Notebook

This jupyter notebook is used to evaluate and compare the metrics and performance of the FIDN model and the CA model. It has the following functions.

- Load detailed data from the wildfires in the test set and store it in a csv file.
- Reads the FIDN model and outputs the performance matrics after performing predictions in the test set.
- Reads the output of the CA model, scales it to the same dimensions as the FIDN and outputs a performance matrics.
- Scaling the output of the FIDN model to the actual geographic space and overlaying it with the forest density information
- Scaling the output of the CA model to the actual geographic space and overlaying it with the forest density information

## Contact

**Bo Pang**

- bo.pang21@imperial.ac.uk
- bo.pang20@outlook.com

**Sibo Cheng**

- sibo.cheng@imperial.ac.uk
