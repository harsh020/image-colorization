import os
import pathlib


BASE_DIR = pathlib.Path('.')
APP_DIR = BASE_DIR/'colorizer'

# MACHINE CONFIG
DEVICE = 'auto' # one of ['cuda', 'cpu', 'auto']

# DATASET VARIABLES
TRAIN_DATASET_PATH = str(APP_DIR/'dataset'/'train')
VALID_DATASET_PATH = None
BATCH_SIZE = 32
SHUFFLE = True

# MODEL VARIABLES
MODEL = 'colornet18' # ['colornet18', 'constantnet']
PRETRINED_BACKBONE = True
FREEZE = True
STATE_DICT_PATH = str(APP_DIR/'serialized'/'colornet18_ckpt.pth')

# TRAINER VARIABLES
EPOCHS = 30
LOSS = 'mse'
METRICS = ['mse', 'channelwise_l2', 'psnr']
OPTIMIZER = 'adam'
LR = 1e-3

# LR SCHEDULER - STEPDECAY
STEP_SIZE = 1000
GAMMA = 0.9

# INFERENCE VARIABLES
SAVE_PATH = str(APP_DIR/'output'/'colored.jpg')
