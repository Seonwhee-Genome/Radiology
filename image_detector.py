import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import numpy as np
from glob import glob
import skimage.io as io

def read_image_into_ndArray(imagefile, PlugIn):
    imageArray = io.imread(imagefile, plugin=PlugIn)
    print("The dimension of the image is ", imageArray.shape)
    return imageArray

def sitk_tile_vec(lstImgs):
    lstImgToCompose = []
    for idxComp in range(lstImgs[0].GetNumberOfComponentsPerPixel()):
        lstImgToTile = []
        for img in lstImgs:
            lstImgToTile.append(sitk.VectorIndexSelectionCast(img, idxComp))
        lstImgToCompose.append(sitk.Tile(lstImgToTile, (len(lstImgs), 1, 0)))
    sitk_show(sitk.Compose(lstImgToCompose))

def sitk_show(img, title=None, margin=0.0, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    #spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    #extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()
    
def figure_images(imageList, z):    
    i = 1
    for a_file in imageList:
        sliced = sitk.GetArrayViewFromImage(a_file[1])[z,:,:]        
        #plt.subplot(round(len(imageList)/3)+1,3, i)
        plt.figure()
        plt.title(a_file[0])
        plt.imshow(sliced, cmap=plt.cm.Greys_r)
        plt.axis('off')
        i = i+1