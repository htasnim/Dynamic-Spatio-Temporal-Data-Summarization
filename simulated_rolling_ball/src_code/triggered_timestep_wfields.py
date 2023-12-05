# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:12:37 2023

@author: htasnim
"""
import matplotlib.pyplot as plt
import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
from multiprocessing import Pool
from vtk.util import numpy_support
import pickle
import pandas as pd

import datetime
begin_time = datetime.datetime.now()
import re
import os.path
from os import path
import scipy.stats
from scipy.stats import chi2_contingency
import cmath
import random
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
import json
from scipy.special import expit
from tkinter import Tcl
import shutil

#############################################

def read_vti(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

#Computation of SMI
def compute_specific_mutual_information(Array1,Array2,ArrayComb,numSamples,bins):

    I11 = np.zeros(bins)
    I12 = np.zeros(bins)
    I21 = np.zeros(bins)
    I22 = np.zeros(bins)
    I31 = np.zeros(bins)
    I32 = np.zeros(bins)

    prob_of_x_given_y=0.0
    prob_of_y_given_x=0.0
    prob_of_x=0.0
    prob_of_y=0.0

    for i in range(0,bins):
        for j in range(0,bins):
            if Array1[i] == 0:
                prob_of_y_given_x=0
            else:
                prob_of_y_given_x = float(ArrayComb[i][j]) / float(Array1[i])

            prob_of_y = float(Array2[j]) / numSamples

            if prob_of_y_given_x != 0 and prob_of_y != 0:
                I11[i] =  I11[i] + prob_of_y_given_x * np.log2(prob_of_y_given_x / prob_of_y)
                
            if prob_of_y_given_x != 0:
                I21[i] = I21[i] + prob_of_y_given_x * np.log2(prob_of_y_given_x)
                
            if prob_of_y != 0:
                I21[i] =  I21[i] - prob_of_y * np.log2(prob_of_y)
                
            if(Array2[i] == 0):
                prob_of_x_given_y = 0
                
            else:
                prob_of_x_given_y = float(ArrayComb[j][i]) / Array2[i]; 

            prob_of_x = float(Array1[j]) / numSamples

            if prob_of_x_given_y != 0 and prob_of_x != 0:
                I12[i] = I12[i] + prob_of_x_given_y * np.log2(prob_of_x_given_y / prob_of_x)

            if(prob_of_x_given_y != 0):
                I22[i] = I22[i] + prob_of_x_given_y * np.log2(prob_of_x_given_y)

            if(prob_of_x != 0):
                I22[i] = I22[i] - prob_of_x * np.log2(prob_of_x)

            if(prob_of_y_given_x > 1.0):
                print("Value of prob_of_y_given_x is greater than 1")

            if(prob_of_x_given_y > 1.0):
                print("Value of prob_of_x_given_y is greater than 1")

    for i in range(0,bins):
        for j in range(0,bins):
            if Array1[i] == 0:
                prob_of_y_given_x=0
            else:
                prob_of_y_given_x = float(ArrayComb[i][j]) / Array1[i]

            prob_of_y = float(Array2[j]) / numSamples

            I31[i] = I31[i] + prob_of_y_given_x * I22[j]

            if(Array2[i] == 0):
                prob_of_x_given_y = 0
            else:
                prob_of_x_given_y = float(ArrayComb[j][i]) / Array2[i] 

            prob_of_x = float(Array1[j]) / numSamples
            I32[i] = I32[i] + prob_of_x_given_y * I21[j]
            
    return I11,I12,I21,I22,I31,I32

#############################################################################################
##Normalizing with bin value
def intensityMap(arr,numBins):
    
    mapped_arr = (numBins-1)*(np.divide((arr - np.min(arr)),np.ptp(arr)))
    
    return mapped_arr.astype(int)

#normalization between 0 - 1
def normalizedScale(arr):
    
    mapped_arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr))
    np.seterr(divide='ignore', invalid='ignore')
    return mapped_arr

# non-linear exponentional function (2D array)
def exponl_func(arr, order):
    mapped_arr = np.zeros(arr.shape)
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            x = arr[i][j]
            if x==0:
                mapped_arr[i][j] =0
            # mapped_arr[i][j] = (np.exp(x)-1) / (np.exp(1)-1)
            else:
                power =  1-np.power(x,-order)
                mapped_arr[i][j] = np.exp(power)
    
    
    return mapped_arr

#nonlinear exponential function for plot generation (1D array)
def exponl_plt(arr, order):
    mapped_arr = np.zeros(arr.shape)
    
    for i in range(arr.shape[0]):
        x = arr[i]
        if x==0:
            mapped_arr[i] =0
        else:
            power =  1-np.power(x,-order)
            mapped_arr[i] = np.exp(power)

    
    # mapped_arr = np.exp(1-np.power(arr,-2))
    
    return mapped_arr
 
#create I1 field
def createI1Field(arr1, arr2, I11, I12, numBins):
    
    I11_field = np.zeros(arr1.shape)
    I12_field = np.zeros(arr2.shape)
    
    arr1=  intensityMap(arr1,numBins)
    arr2=  intensityMap(arr2,numBins)
    
    
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            I11_field[i][j] = I11[arr1[i][j]] 
            I12_field[i][j] = I12[arr2[i][j]]
    
    return I11_field, I12_field
    
#create I2 field
def createI2Field(arr1, arr2, I21, I22, numBins):
    
    I21_field = np.zeros(arr1.shape)
    I22_field = np.zeros(arr2.shape)
    
    arr1=  intensityMap(arr1,numBins)
    arr2=  intensityMap(arr2,numBins)
    
    
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            I21_field[i][j] = I21[arr1[i][j]] 
            I22_field[i][j] = I22[arr2[i][j]]
    
    return I21_field, I22_field

#create I3 field (Use for later)
def createI3Field(arr1, arr2, I31, I32, numBins):
    
    I31_field = np.zeros(arr1.shape)
    I32_field = np.zeros(arr2.shape)
    
    arr1=  intensityMap(arr1,numBins)
    arr2=  intensityMap(arr2,numBins)
    
    
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            I31_field[i][j] = I31[arr1[i][j]] 
            I32_field[i][j] = I32[arr2[i][j]]
    
    return I31_field, I32_field

#create PMI field
def compute_pointwise_mutual_information(Array1,Array2,ArrayComb,numSamples,bins):
    pmi_array = np.zeros_like(ArrayComb)
    
    for i in range(bins):
        for j in range(bins):
            
            if ArrayComb[i][j]==0:
                pmi_array[i][j]=0
            elif ArrayComb[i][j] > 0 and Array1[i] > 0  and Array2[j] > 0:
                prob_x = Array1[i]/float(numSamples)
                prob_y = Array2[j]/float(numSamples)
                joint_prob_xy = ArrayComb[i][j]/float(numSamples)
                pmi_array[i][j] = np.log2(joint_prob_xy/(prob_x*prob_y))
            else:
                pmi_array[i][j]=0
             
            ## normalize betweem [-1,1]
            #if pmi_array[i][j] != 0:
            #    pmi_array[i][j] = pmi_array[i][j]/(-np.log2(joint_prob_xy))
                
    return pmi_array

#################################################################################
## Compute PMI field
#################################################################################

def createPMIField(arr1, arr2, pmi_array, numBins):

    pmi_field = np.zeros(arr1.shape)
    
    arr1=  intensityMap(arr1,numBins)
    arr2=  intensityMap(arr2,numBins)
    
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            pmi_field[i][j] = pmi_array[arr1[i][j]][arr2[i][j]] 
    
    return pmi_field 

def createFusionImage(field1, field2,Ifield1,Ifield2,timestep_fuse,time,confidence_th):
    
    fused_field_data =  np.zeros(field1.shape)
    fused_field_I1 = np.zeros(field1.shape)
    
    # pmi_fused_field =  np.zeros(field1.shape)
   
        
    for i in range(field1.shape[0]):
        for j in range(field1.shape[1]):
            if (Ifield1[i][j]> Ifield2[i][j]):
                fused_field_data[i][j] = field1[i][j]
                fused_field_I1[i][j] = Ifield1[i][j]
                if time == 1 :
                    timestep_fuse[i][j] = time
            else:
                fused_field_data[i][j] = field2[i][j]
                fused_field_I1[i][j] = Ifield2[i][j]
                timestep_fuse[i][j] = time+1
                   
            
        
    for i in range(field1.shape[0]):
        for j in range(field1.shape[1]):
            # for k in range(field1.shape[2]):
            if fused_field_data[i][j]<confidence_th:
                timestep_fuse[i][j] = 0
                

    return fused_field_data,fused_field_I1,timestep_fuse


def maxFusedField(field1, field2):
    
    fused_field =  np.zeros(field1.shape)
    for i in range(field1.shape[0]):
        for j in range(field1.shape[1]):
            if (field1[i][j] > field2[i][j]):
                fused_field[i][j] = field1[i][j]
            else: 
                fused_field[i][j] = field2[i][j]
    
    return fused_field

def weightedFusedField(I1_arr1_2d, arr2_2d,I11_field,I12_field):
    
     return np.divide(np.add(np.multiply(I1_arr1_2d,I11_field),np.multiply(arr2_2d,I12_field)), np.add(I11_field,I12_field))
    
  

    

## Store files in vtk format

def write_vti(filename,data):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update() 

def store_vtk_format(data,arr_name,fname):
    all_values = np.asarray(data).flatten()
    all_vals_3D = numpy_to_vtk(all_values)
    all_vals_3D.SetName(arr_name)
    
    # print (all_vals_3D.GetNumberOfTuples())
    
    dataset = vtk.vtkImageData()
    # dataset.SetDimensions(data.shape[0],data.shape[1],data.shape[2])
    dataset.SetDimensions(data.shape[0],data.shape[1],1)
    dataset.SetSpacing(1.0, 1.0, 1.0)
    dataset.GetPointData().AddArray(all_vals_3D)
    
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(dataset)
    writer.Write()

def convert_arr_to_vti(data,arr_name):
    dataSet = vtk.vtkImageData()
    dataSet.SetDimensions(1,data.shape[1],data.shape[0])
    data = np.fliplr(data)
    dataSet.SetSpacing([1,1,1])
    dataSet.SetOrigin([0,0,0])
    vtk_arr = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName(arr_name)
    dataSet.GetPointData().AddArray(vtk_arr)
    return dataSet

def segment_feature(fname,confidence_th,size_threshold, tstep):
    
    data = read_vti(fname)
    data.GetPointData().SetActiveScalars('ImageFile')    
    # gbounds = data.GetBounds()

    thresholding = vtk.vtkThreshold()
    # thresholding.ThresholdByLower( confidence_th )
    thresholding.ThresholdByUpper( confidence_th )
    thresholding.SetInputData(data)
    
    seg = vtk.vtkConnectivityFilter()
    seg.SetInputConnection(thresholding.GetOutputPort())
    seg.SetExtractionModeToLargestRegion()
    seg.Update()

    segmentation = vtk.vtkConnectivityFilter()
    segmentation.SetInputConnection(thresholding.GetOutputPort())
    segmentation.SetExtractionModeToAllRegions()
    segmentation.ColorRegionsOn()
    segmentation.Update()

    ug = segmentation.GetOutput()
    num_segments = segmentation.GetNumberOfExtractedRegions()
    
    # print(ug)
    # print(num_segments)
    
    ## compute volumes of each bubble:
    bubble_volumes = np.zeros(num_segments)
    for i in range(ug.GetPointData().GetArray('RegionId').GetNumberOfTuples()):
        regionId = int(ug.GetPointData().GetArray('RegionId').GetTuple(i)[0])
        bubble_volumes[regionId] = bubble_volumes[regionId]+1     
    
    count = 0
    for i in range(num_segments):
        if  bubble_volumes[i] > size_threshold:
            count = count + 1
            
            
    return count

def checkorcreatepath(givenpath):
    if not os.path.isdir(givenpath):
        try:
            os.makedirs(givenpath)
        except OSError:
            print ("Creation of the directory %s failed" % givenpath)
        else:
            print ("Successfully created the directory %s " % givenpath)
    return givenpath

def fusion_timesteps(I1_temp_fuse, timestep_fuse,var1, var2,t1,t2, output_path,time,confidence_th):
    numBins = 128
    
    I1_arr1_2d = I1_temp_fuse
    I1_arr1 = I1_temp_fuse.flatten()

    
    arr1_2d = var1
    arr1 = var1.flatten()
    
    arr2_2d = var2
    arr2 = var2.flatten()


    numSamples= I1_arr1.shape[0]
    # print(numSamples)

    
    Array  = np.histogram(arr1,bins= numBins)[0] 
    Array1 = np.histogram(I1_arr1,bins= numBins)[0] 
    Array2 = np.histogram(arr2,bins= numBins)[0]

    ArrayComb = np.histogram2d(I1_arr1, arr2, bins=numBins)[0]
    ArrayComb_field = np.histogram2d(arr1, arr2, bins=numBins)[0]
    
    
    # print(ArrayComb.shape)

    # #SMI function call
    I11,I12,I21,I22,I31,I32 = compute_specific_mutual_information(Array1,Array2,ArrayComb,numSamples,numBins)
    
    I11f,I12f,I21f,I22f,I31f,I32f = compute_specific_mutual_information(Array,Array2,ArrayComb_field,numSamples,numBins)
    
    
    I11_field, I12_field = createI1Field(I1_arr1_2d, arr2_2d, I11, I12, numBins)
    I11_field_i, I12_field_i = createI1Field(arr1_2d, arr2_2d, I11f, I12f, numBins)
    
    
    
    
    #saving I1 field
    opath1 = checkorcreatepath(output_path + 'I1_Field_individual/')
    opath2 = checkorcreatepath(output_path + 'I1_Field_fused/')
    arr_name = 'I1_val'
    fname = opath1 + 'I1_field_X_t' + str(t1)  + '_Y_t' + str(t2) + '.vti'
    store_vtk_format(I11_field_i, arr_name, fname)
    fname = opath1 + 'I1_field_X_t' + str(t2)  + '_Y_t' + str(t1) + '.vti'
    store_vtk_format(I12_field_i, arr_name, fname)
    
    fname = opath2 + 'I1_field_X_t' + str(t1)  + '_Y_t' + str(t2) + '.vti'
    store_vtk_format(I11_field, arr_name, fname)
    fname = opath2 + 'I1_field_X_t' + str(t2)  + '_Y_t' + str(t1) + '.vti'
    store_vtk_format(I12_field, arr_name, fname)
    
    
    I1_fused_data_field, I1_fused_I1_field, timestep_fuse = createFusionImage(I1_temp_fuse, arr2_2d,I11_field,I12_field,timestep_fuse,time,confidence_th)
   
    return  I1_fused_data_field, I1_fused_I1_field, timestep_fuse


def vti_to_numpy(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
   
    dims = reader.GetOutput().GetDimensions()
    
       
    arr = reader.GetOutput().GetPointData().GetArray('ImageFile')
    numpy_array = vtk_to_numpy(arr)
    data = np.reshape(numpy_array,dims)
    return data

#################################################################################

## Parameters
###############
dim_x = 800
dim_y = 400
var_name = 'sample_ball_ViBe'
init_time = 0     
end_time = 10 

input_path = "rolling_ball_input/" 


## similarity threshold for segmentation
confidence_th = 255

fused_time=15

## feature size threshold. The value indicates number of voxels. Any feature below this threshold will be discarded.
size_threshold = 1500

bubble_counts = np.zeros(end_time - init_time+1)
#################################################################################

# Run for all timesteps
for ii in range(init_time,end_time+1):
    # print ('Processing time step: ' + str(ii))
    inpfname = input_path + var_name + str(ii) + '.vti'
    bubble_counts[ii-init_time]=segment_feature(inpfname,confidence_th,size_threshold, ii)
    print(ii)
    


for root, dirs, files in os.walk(input_path):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        varfiles =list(Tcl().call('lsort', '-dict', files))

output_path = checkorcreatepath("Output/threshold"+str(confidence_th)+"/") 

opath = checkorcreatepath(output_path + "Final_fused_images/")


ii=0
# to be consistent with naming
start_time = 0
while ii < end_time-init_time+1:
    if ii == end_time-init_time:
        inpfname = input_path + var_name + str(ii) + '.vti'
        shutil.copy(inpfname, opath)
    else:
        if bubble_counts[ii] != bubble_counts[ii+1]:
            inpfname = input_path + var_name + str(ii) + '.vti'
            shutil.copy(inpfname, opath)
            
        elif bubble_counts[ii] == bubble_counts[ii+1]:
            if ii+2 < end_time-init_time+1 and bubble_counts[ii+1] == bubble_counts[ii+2] :                
                count = 1
                while ii + count + 1 < end_time-init_time +1 and bubble_counts[ii+count] == bubble_counts[ii + count + 1]:
                    count = count + 1
                
                count = count + 1
                print('count =' + str(count))
                inputfile = input_path + var_name + str(ii+init_time) + '.vti'
                I1_temp_fuse = vti_to_numpy(inputfile)
                I1_fuse_field = np.zeros(I1_temp_fuse.shape)
                timestep_fuse = np.zeros(I1_temp_fuse.shape)
                time  = 1
         
                
                for jj in range(ii, ii+count-1):
                    file_path = input_path + var_name + str(jj+init_time) + '.vti'
                    var1 = vti_to_numpy(file_path)
                    inputfile = input_path + var_name  + str(jj+1+init_time) + '.vti' 
                    var2 = vti_to_numpy(inputfile)
                    I1_temp_fuse, I1_fuse_field, timestep_fuse= fusion_timesteps(I1_temp_fuse,timestep_fuse,var1, var2, jj, jj + 1, output_path,time,confidence_th)
                    time = time+1
                    
                I1path = checkorcreatepath(opath+"I1_fused_data/")
                oname = I1path + var_name + str(ii)+ 'to' +str(ii+count-1) + '.vti'
                store_vtk_format(I1_temp_fuse, 'ImageFile', oname)
                    
                timefuse_path= checkorcreatepath(opath+"time_fused/")
                oname = timefuse_path + var_name + str(ii)+ 'to' +str(ii+count-1) + '.vti'
                store_vtk_format(timestep_fuse, 'ImageFile', oname)
                
                
                field_path = checkorcreatepath(opath+'I1_fused_field/')
                oname = field_path + var_name + str(ii)+ 'to' +str(ii+count-1) + '.vti'
                store_vtk_format(I1_fuse_field,  'ImageFile', oname)
            
                ii = ii + count -1
            else:
                inpfname = input_path + var_name + str(ii) + '.vti'
                shutil.copy(inpfname, opath)
                print('skipping timestep'+ str(ii+1))
                # print('saving' + str(bubble_counts[ii]) + 'in' +str(ii) + 'skipping' + str(bubble_counts[ii+1]) + 'in' + str(ii+1) )
                ii=ii+1
    
    print(str(ii))
    ii = ii+1
            
print ('done processing all time steps')


