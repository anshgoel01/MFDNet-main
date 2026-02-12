#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()

        # ---------------------------------------------------
        # SYSTEM
        # ---------------------------------------------------
        self._C.GPU = [0]
        self._C.VERBOSE = True

        # ---------------------------------------------------
        # MODEL
        # ---------------------------------------------------
        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'RainDS_MFDNet'

        # ---------------------------------------------------
        # OPTIMIZATION
        # ---------------------------------------------------
        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 4          # ✅ Better for RainDS
        self._C.OPTIM.NUM_EPOCHS = 100        # ✅ Match deraining training
        self._C.OPTIM.NEPOCH_DECAY = [200]    # LR decay after 200 epochs
        self._C.OPTIM.LR_INITIAL = 1e-4       # ✅ Stable for restoration
        self._C.OPTIM.LR_MIN = 1e-6
        self._C.OPTIM.BETA1 = 0.9

        # ---------------------------------------------------
        # TRAINING
        # ---------------------------------------------------
        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 5
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False

        # ✅ IMPORTANT — dataset root (NOT train_set)
        self._C.TRAINING.TRAIN_DIR = r"D:/capstone/RainDS/RainDS_real"
        self._C.TRAINING.VAL_DIR   = r"D:/capstone/RainDS/RainDS_real"

        self._C.TRAINING.SAVE_DIR = "checkpoints"

        # Patch sizes
        self._C.TRAINING.TRAIN_PS = 128       # ✅ Standard deraining patch
        self._C.TRAINING.VAL_PS = 128      # ✅ Full resolution validation

        # ---------------------------------------------------
        # CHECKPOINTING (NEW BUT SAFE)
        # ---------------------------------------------------
        self._C.TRAINING.SAVE_AFTER_EVERY = 10   # ✅ Save every 10 epochs

        # ---------------------------------------------------
        # LOAD YAML OVERRIDES
        # ---------------------------------------------------
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
