from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEVICE = 'cpu'
_C.PATH = './data/'
_C.BACKBONE = 'mobilenet_v2'
_C.OUTPUT_DIR = './test'
_C.MODEL_DIR = './pth'
_C.LOG_DIR = './log'



_C.IMAGE_SIZE = 224
_C.EPOCH = 100
_C.TRAIN_BATCH = 16
_C.VAL_BATCH = 1
_C.WEIGHT = 1
_C.BIN = 4
_C.JITTER = 6
_C.OVERLAPPING = 0.1

