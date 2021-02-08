import seeed_mlx9064x
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

#mlx = seeed_mlx9064x.grove_mxl90641()
mlx = seeed_mlx9064x.grove_mxl90640()
mlx.refresh_rate = seeed_mlx9064x.RefreshRate.REFRESH_8_HZ  # The fastest for raspberry 4 
mlx_shape = (24,32)

mlx_interp_val = 10
mlx_interp_shape = (mlx_shape[0]*mlx_interp_val,
                    mlx_shape[1]*mlx_interp_val) #new shape(240 by 320)

fig = plt.figure(figsize=(4,3))#start fig??
ax = fig.add_subplot(111)#subplot?
fig.subplots_adjust(0,0,1,1) # get rid of unnecessary padding
#fig.subplots_adjust(0.002,0.0001,0.9,1)
#fig.subplots_adjust(0.002,0.02,1,1) # get rid of unnecessary padding
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

##------------------------------HSV SETUP----------------------------------------------------
def hsv_color_space(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L-H","HSV1 BARS")
    l_s = cv2.getTrackbarPos("L-S","HSV1 BARS")
    l_v = cv2.getTrackbarPos("L-V","HSV1 BARS")
    u_h = cv2.getTrackbarPos("U-H","HSV1 BARS")
    u_s = cv2.getTrackbarPos("U-S","HSV1 BARS")
    u_v = cv2.getTrackbarPos("U-V","HSV1 BARS")
    lower_color = np.array([l_h,l_s,l_v])
    upper_color= np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv,lower_color,upper_color)
    return mask

def houghLines(img):
    #threshold1 = cv2.getTrackbarPos("T1","HOUGH1 BARS")
    #threshold2 = cv2.getTrackbarPos("T2","HOUGH1 BARS")
    edges = cv2.Canny(img,100,100)
    return edges

def nothing(X):
    pass

def createHSVwindow():
    cv2.namedWindow("HSV1 BARS")
    cv2.createTrackbar("L-H","HSV1 BARS",15,180, nothing)
    cv2.createTrackbar("L-S","HSV1 BARS",0,255, nothing)
    cv2.createTrackbar("L-V","HSV1 BARS",25,180, nothing)
    cv2.createTrackbar("U-H","HSV1 BARS",68,180, nothing)
    cv2.createTrackbar("U-S","HSV1 BARS",83,255, nothing)
    cv2.createTrackbar("U-V","HSV1 BARS",150,255, nothing)
##-------------------------------------------------------------------------------------------------------
    


createHSVwindow()
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
        graph_image = hsv_color_space(graph_image)
        #graph_image = houghLines(graph_image)
        #graph_image = crop_image(graph_image, pixel_value=255)
        #print("shape: ",graph_image.shape)
        cv2.imshow("graph image",graph_image)
    except:
        continue
    # approximating frame rate
    t_array.append(time.monotonic()-t1)
    if len(t_array)>10:
        t_array = t_array[1:] # recent times for frame rate approx
    print('Frame Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))


