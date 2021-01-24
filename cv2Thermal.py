import seeed_mlx9064x
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

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
#frame = np.tile(0,192)
mlx_shape = (16,12)

mlx_interp_val = 10
mlx_interp_shape = (mlx_shape[0]*mlx_interp_val,
                    mlx_shape[1]*mlx_interp_val) #new shape(160 by 120)

fig = plt.figure(figsize=(12,9))#start fig??
ax = fig.add_subplot(111)#subplot?
fig.subplots_adjust(0.05,0.05,0.95,0.95) # get rid of unnecessary padding
therm1 = ax.imshow(np.zeros(mlx_interp_shape),interpolation='none',
                   cmap=plt.cm.bwr,vmin=25,vmax=45) # preemptive image
fig.canvas.draw() # draw figure to copy background
ax_background = fig.canvas.copy_from_bbox(ax.bbox) # copy background

frame = np.zeros(mlx_shape[0]*mlx_shape[1])

def updateplot():
    fig.canvas.restore_region(ax_background) # restore background
    mlx.getFrame(frame) # read mlx90640
    data_array = np.fliplr(np.reshape(frame,mlx_shape)) # reshape, flip data
    data_array = ndimage.zoom(data_array,mlx_interp_val) # interpolate
    therm1.set_array(data_array) # set data
    therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
    ax.draw_artist(therm1) # draw new thermal image
    fig.canvas.blit(ax.bbox) # draw background
    fig.canvas.flush_events() # show the new image
    return

def crop_image(image, pixel_value=255):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
    crop_rows = gray[~np.all(gray == pixel_value, axis=1), :]
    cropped_image = crop_rows[:, ~np.all(crop_rows == pixel_value, axis=0)]
    return cropped_image

t_array = []
while True:
    t1 = time.monotonic() # for determining frame rate
    try:
        updateplot() # update plot
        canvas = FigureCanvas(fig)
        plt.axis('off')
        plt.axis('tight')
        plt.axis('image')
        canvas.draw()
        graph_image = np.array(fig.canvas.get_renderer()._renderer)
        graph_image = crop_image(graph_image, pixel_value=255)
        graph_image = cv2.cvtColor(graph_image,cv2.COLOR_GRAY2RGB)
        cv2.imshow("graph image",graph_image)
    except:
        continue
    # approximating frame rate
    t_array.append(time.monotonic()-t1)
    if len(t_array)>10:
        t_array = t_array[1:] # recent times for frame rate approx
    print('Frame Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))


