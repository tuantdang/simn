

# @File      config.py
# @Author    Tuan Dang  
# @Email    tuan.dang@uta.edu or dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

''' Description
The config class provides an easy method  to map the YAML key value into object properties/values
without explicitly declaring the properties and assign them.
This helps coding concise and consitantly over the configuration.
Example:
YAML:
    dataset: # Dataset path & settings
    name: tum
    root: /home/tuandang/workspace/datasets/tum/extract
    sub_path: rgbd_dataset_freiburg1_desk
    cached: True
    multi_processing: False
    chunk_size: 100 # multi-processing

    # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    camera: # freiburg1
    width: 640
    height: 480
    fx: 517.3
    fy: 516.5
    cx: 318.6
    cy: 255.3
    dscale: 5000.0 # depth scale
    fps: 30

Python:
    cfg = Config(path="config/kitti.yaml").config()
    #cfg.print_config() # Print all config values
    print(cfg.dataset.name) # Print specific mapped property's value
Result:
    kitti
'''

import yaml
import torch
from lib.utils import print_object

class Config:
    def __init__(self, path=None, dictionary=None):
        if dictionary == None and path != None:
            with open(path, 'r') as file:
                dictionary = yaml.safe_load(file)
                # print(dictionary)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(dictionary=value)  # Recursively convert nested dicts
            setattr(self, key, value)
            
    def config(self): # Convert string properties in yaml to specify in Python-based data-types
        self.device = torch.device(self.device)
        self.dtype.point = torch.float32 if self.dtype.point == "torch.float32" else torch.float64
        self.dtype.transformation = torch.float32 if self.dtype.transformation == "torch.float32" else torch.float64
        self.dtype.index = torch.float32 if self.dtype.index == "torch.int32" else torch.int64
        return self

    def print_config(self):
        print_object(self)
        
