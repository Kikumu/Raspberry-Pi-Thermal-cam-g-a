import seeed_mlx9064x
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
import random as randchoice
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import import_ipynb
from ipynb.fs.full.Functions import Pool_
from ipynb.fs.full.Functions import newSpecies
from ipynb.fs.full.Functions import createNewGenome
from ipynb.fs.full.Functions import connectionGene
from ipynb.fs.full.Functions import Neuron
from ipynb.fs.full.Functions import updateInputs
from ipynb.fs.full.Functions import obtainOutputs
from ipynb.fs.full.Functions import setNetworkTrackbarValues
from ipynb.fs.full.Functions import CreateTrackbarValues
from ipynb.fs.full.Functions import createHoughTrackbarValues
from ipynb.fs.full.Functions import setDefaultTrackbarValues
from ipynb.fs.full.Functions import hsv_color_space
from ipynb.fs.full.Functions import evaluateGenome
from ipynb.fs.full.Functions import houghLines

cv2.namedWindow("Default Window")
cv2.namedWindow("Network Window")
cv2.namedWindow("HSV BARS")
cv2.namedWindow("HOUGH BARS")

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

t_array = []

def load_pool():
    pickle_in = open('Pool_data.pickle','rb')
    saved_pool = pickle.load(pickle_in)
    pickle_in.close()
    return saved_pool

Genome_Pool = load_pool()
#store and pick random genome
Total_Population = []
for s in Genome_Pool.species:
    for g in s.genomes:
        Total_Population.append(g)
        
test_subject = randchoice.choice(Total_Population)
#print('tpyre', type(test_subject))
CreateTrackbarValues()
testOutputs = np.array([])
testOutputs = np.tile(0,6)
while True:
    t1 = time.monotonic() # for determining frame rate
    #try:
    updateplot() # update plot
    canvas = FigureCanvas(fig)
    plt.axis("off")
    plt.axis("tight")
    plt.axis("image")
    canvas.draw()
    graph_image = np.array(fig.canvas.get_renderer()._renderer)
    cv2.imshow("graph image",graph_image)
    setDefaultTrackbarValues()
        
    mask   = hsv_color_space(graph_image)
    mask   = cv2.resize(mask, (128,128))
    #print('works')
    updateInputs(test_subject,mask.flatten())
        #print('Genome')
    #print('works1')
    evaluateGenome(test_subject)
    #print('Genome1')
    setNetworkTrackbarValues(obtainOutputs(test_subject,testOutputs))
        
    print(obtainOutputs)
    mask = hsv_color_space(graph_image)
    mask  = houghLines(mask)
    mask  = cv2.resize(mask, (128,128))
        
    cv2.imshow("Network Window", mask)
    #print('Final')
    #except:
        #continue
    # approximating frame rate
    t_array.append(time.monotonic()-t1)
    if len(t_array)>10:
        t_array = t_array[1:] # recent times for frame rate approx
    print('Frame Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))
