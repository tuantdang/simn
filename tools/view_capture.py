# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import cv2
import numpy as np
from os import listdir
from natsort import natsorted
import matplotlib.pyplot as plt
from settings import *


running = True
def signal_handler(signum, frame):
    global running
    running = False
    print(f'\nSIGINT received: signum = {signum}, frame={frame}')
import signal
signal.signal(signal.SIGINT, signal_handler)



filenames = natsorted(listdir(rgb_dir))
count = len(filenames)

# Set up the Matplotlib figure
plt.ion()  # Interactive mode on
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for id in range(count):
# for id in range(1100, 1700):
    rgb_fn = f'{rgb_dir}/{id:04d}.npy'
    depth_fn = f'{depth_dir}/{id:04d}.npy'
    rgb = np.load(rgb_fn)
    depth = np.load(depth_fn)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    print(f'mean depth = {np.mean(depth)}')
    
    # depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    # depth_img = np.uint8(depth_img)

    # depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)    #[w,h,3]
    # frame = np.concatenate([rgb, depth_colored], axis=1)
    # cv2.imshow(saved_dir, frame)
    # cv2.waitKey(fps)  

    #  Clear previous images
    ax[0].cla()
    ax[1].cla()

    # Display images
    ax[0].imshow(rgb)
    ax[0].set_title(f"Color Image: {id:04d}")
    # ax[0].axis("off")

    ax[1].imshow(depth, cmap="jet")
    ax[1].set_title(f"Depth Image {id:04d}")
    # ax[1].axis("off")
    # plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.savefig(f'{png_dir}/{id:004d}.png')

    # plt.pause(1.0/fps)  # Pause for a short moment to update the figure
    plt.pause(0.0001)
    if not running:
        break