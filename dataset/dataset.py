# @File      config.py
# @Author    Tuan Dang  
# @Email    tuan.dang@uta.edu or dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved

''' Description
Provide generic interface for specific datasets or sensors which need to implement the required methods
'''


import torch
from lib.logger import Logger

class Dataset:
    def __init__(self, cfg, logger:Logger=None):
        self.cfg = cfg
        self.logger = logger
                
        # Shared properties arcorss all classes
        self.frame_id = -1
        self.pc_count = 10000
    
    def available(self): # Query available data
        pass

        
    def next_frame(self): # Get next frame: frame_id, pc, pc_registration
       pass
   
