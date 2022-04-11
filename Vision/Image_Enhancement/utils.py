import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def draw_cicle(shape,diamiter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar

    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)


def filter_circle(TFcircleIN,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)


def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)

    
def make_Lf_Hf_decomposition(path):
    img = plt.imread(path)# /float(2**8)

    shape = img.shape[:2]

    TFcircleIN   = draw_cicle(shape=img.shape[:2],diamiter=50)
    TFcircleOUT  = ~TFcircleIN
    fft_img = np.zeros_like(img,dtype=complex)

    for ichannel in range(fft_img.shape[2]):
        fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))

    fft_img_filtered_IN = []
    fft_img_filtered_OUT = []
    ## for each channel, pass filter
    for ichannel in range(fft_img.shape[2]):
        fft_img_channel  = fft_img[:,:,ichannel]
        ## circle IN
        temp = filter_circle(TFcircleIN,fft_img_channel)
        fft_img_filtered_IN.append(temp)
        ## circle OUT
        temp = filter_circle(TFcircleOUT,fft_img_channel)
        fft_img_filtered_OUT.append(temp) 

    fft_img_filtered_IN = np.array(fft_img_filtered_IN)
    fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
    fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
    fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))
    

    img_reco              = inv_FFT_all_channel(fft_img)
    img_reco_filtered_IN  = inv_FFT_all_channel(fft_img_filtered_IN)
    img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)

    fig = plt.figure(figsize=(25,18))
    ax  = fig.add_subplot(1,3,1)
    ax.imshow(np.abs(img_reco))
    ax.set_title("original image")

    ax  = fig.add_subplot(1,3,2)
    ax.imshow(np.abs(img_reco_filtered_IN))
    ax.set_title("low pass filter image")


    ax  = fig.add_subplot(1,3,3)
    ax.imshow(np.abs(img_reco_filtered_OUT))
    print(np.min(np.abs(img_reco_filtered_IN)), np.max(np.abs(img_reco_filtered_IN)))
    ax.set_title("high pass filtered image")
    plt.show()