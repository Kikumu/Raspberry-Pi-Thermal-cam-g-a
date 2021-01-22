import seeed_mlx9064x
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage 
#mlx = seeed_mlx9064x.grove_mxl90640()
mlx = seeed_mlx9064x.grove_mxl90641()
mlx.refresh_rate = seeed_mlx9064x.RefreshRate.REFRESH_8_HZ  # The fastest for raspberry 4 
# REFRESH_0_5_HZ = 0b000  # 0.5Hz
# REFRESH_1_HZ = 0b001  # 1Hz
# REFRESH_2_HZ = 0b010  # 2Hz
# REFRESH_4_HZ = 0b011  # 4Hz
# REFRESH_8_HZ = 0b100  # 8Hz
# REFRESH_16_HZ = 0b101  # 16Hz
# REFRESH_32_HZ = 0b110  # 32Hz
# REFRESH_64_HZ = 0b111  # 64Hz
#frame = [0]*192
frame = np.tile(0,192)
mlx_shape = (16,12)
plt.ion()
fig,ax = plt.subplots(figsize=(12,7))
therm1 = ax.imshow(np.zeros(mlx_shape),vmin=0,vmax=60)
cbar = fig.colorbar(therm1)
cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14) 
frame = np.zeros((16*12,))
t_array = []
while True:
    t1 = time.monotonic()
    try:
        #frame = [0]*768
        mlx.getFrame(frame)
        data_array = (np.reshape(frame,mlx_shape)) # reshape to 16x12
        data_array = ndimage.interpolation.zoom(data_array,zoom=10)
        therm1.set_data(np.fliplr(data_array)) # flip left to right
        therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
        cbar.on_mappable_changed(therm1) # update colorbar range
        plt.pause(0.001) # required

        t_array.append(time.monotonic()-t1)
        print('Sample Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))
        #print("Drawing frame..",frame[0])
    except ValueError:
        print("Fail")
        continue
