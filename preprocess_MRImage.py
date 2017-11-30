import cv2
import numpy as np
import dicom as pdicom
import os
import subprocess as sp
import SimpleITK as sitk
import skimage.io as io

class Four_modalities(object):
    def __init__(self, T1, T1c, T2, FLAIR, GroundTruth):
        self.img_T1 = T1
        self.img_T2 = T2
        self.img_T1c = T1c
        self.img_FLAIR = FLAIR
        self.GT = GroundTruth
        
    def smoothing_Imgs(self):
        img_T1_Smooth = sitk.CurvatureFlow(self.image1=self.img_T1, timeStep=0.125, numberOfIterations=5)
        img_T2_Smooth = sitk.CurvatureFlow(self.image1=self.img_T2, timeStep=0.125,  numberOfIterations=5)
        img_T1c_Smooth = sitk.CurvatureFlow(self.image1=self.img_T1c, timeStep=0.125, numberOfIterations=5)
        img_FLAIR_Smooth = sitk.CurvatureFlow(self.image1=self.img_FLAIR, timeStep=0.125, numberOfIterations=5)
        GT_Smooth = sitk.CurvatureFlow(self.image1=self.GT, timeStep=0.125, numberOfIterations=5)
        return (img_T1_Smooth, img_T1c_Smooth, img_T2_Smooth, img_FLAIR_Smooth, GT_Smooth)
    
    def rescale_Imgs(self):
        (img_T1_Smooth, img_T1c_Smooth, img_T2_Smooth, img_FLAIR_Smooth, GT_Smooth) = self.smoothing_Imgs()
        img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1_Smooth), sitk.sitkUInt8)
        img_T2_255 = sitk.Cast(sitk.RescaleIntensity(img_T2_Smooth), sitk.sitkUInt8)
        img_FLAIR_255 = sitk.Cast(sitk.RescaleIntensity(img_FLAIR_Smooth), sitk.sitkUInt8)
        img_T1c_255 = sitk.Cast(sitk.RescaleIntensity(img_T1c_Smooth), sitk.sitkUInt8)
        GT_255 = sitk.Cast(sitk.RescaleIntensity(GT_Smooth), sitk.sitkUInt8)
        return (img_T1_255, img_T1c_255, img_T2_255, img_FLAIR_255, GT_255)