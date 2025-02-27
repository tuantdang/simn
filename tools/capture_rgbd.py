# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import numpy as np
import time
import pyrealsense2 as rs   
import signal
import cv2
import os
import shutil
from settings import *

r = input('Capture?')
if r != 'y':
    exit()

running = True

def signal_handler(signum, frame):
    global running
    running = False
    print(f'\nSIGINT received: signum = {signum}, frame={frame}')


print(saved_dir)
print(rgb_dir)
print(depth_dir)
print(png_dir)

# exit()

if os.path.exists(saved_dir):
    shutil.rmtree(saved_dir)

os.makedirs(saved_dir, exist_ok=True)
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)


signal.signal(signal.SIGINT, signal_handler)
pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

profile = pipeline.start(rs_config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


# Depth sensor
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = rs_config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()
dscale = 1./depth_sensor.get_depth_scale() # 1000
print(dscale)
# exit()

#Skip some first frames
for _ in range(30*3):
    pipeline.wait_for_frames()

# Set up the Matplotlib figure
import matplotlib.pyplot as plt
# plt.ion()  # Interactive mode on
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

frame_id = -1
t1 = time.time()
while running:
    frame_id += 1
    
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    # depth_frame = frameset.get_depth_frame()

    rgb = np.asanyarray(color_frame.get_data())

    # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    #Align
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    aligned_depth_frame = frameset.get_depth_frame()
    depth = np.asanyarray(aligned_depth_frame.get_data()) #[w,h]
    
    

    # plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    # plt.savefig(f'{png_dir}/{frame_id:004d}.png')

    np.save(f'{rgb_dir}/{frame_id:004d}.npy', rgb)
    np.save(f'{depth_dir}/{frame_id:004d}.npy', depth)

    # For debuggin
    colorizer = rs.colorizer()
    depth_frame_view = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())     #[w,h,3]
    frame = np.concatenate([rgb, depth_frame_view], axis=1)
    cv2.imshow("Realsense D435i", frame)
    cv2.waitKey(1)  

    # #  Clear previous images
    # ax[0].cla()
    # ax[1].cla()
    # # Display images
    # ax[0].imshow(rgb)
    # ax[0].set_title("Color Image")
    # ax[1].imshow(depth, cmap="jet")
    # ax[1].set_title("Depth Image")
    # plt.pause(0.0001)  # Pause for a short moment to update the figure
    t2 = time.time()
    t = t2 - t1
    print(f' frame id = {frame_id:04d} -  accsing camera time : {t*1000: 0.2f} ms or fps = {1/t: 0.1f}')
    t1 = t2