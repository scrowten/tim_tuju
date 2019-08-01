# import the necessary packages
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import cv2
import time
import os
from os import listdir
from os.path import isfile, join
from math import sqrt
from get_colort_functions import *


# Iterate files in specific folder
cwd = os.getcwd()
files = []
for i in listdir(cwd + r"\\input"):
    if ".jpg" in i:
        files.append(i)
df_color = pd.read_csv('color_fix.csv', sep=';', index_col=0)
columns_order = ['file_name','top','parent_color','child_color','r','g','b','h','l','s','b_med','b_std','brightness','brightness_lab', 'hue','luminance','intensity','intensity_label','saturation']

bucket = []
print("Debug: generating data")
for file in files:
    x = r'\\input'+ r"\\" + file
    img = cv2.imread(cwd+x)
    try:
        img_resized = cv2.resize(img,(32,32))
    except:
        pass
    try:
        # Get HLS
        img_grey = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HLS)
        h,l,s = get_dominant_black(img_grey, k=5, image_processing_size=None)
        
    except:
        h,l,s = 999, 999, 999
    try:
        # Get Median and Std. Deviation
        med, std = np.median(img_grey),np.std(img_grey)
    except:
        med, std = 999, 999
    try:
        # Get RGB
        dominant_color = get_dominant_color(img_resized, k=5,l=3, image_processing_size = None)
        count = 1
        for a in dominant_color:
            
            df_color["jarak"] = np.vectorize(jarak)(df_color['r'].astype(int), df_color['g'].astype(int),df_color['b'].astype(int), int(a[0]), int(a[1]), int(a[2]))
            
            index_min = df_color.jarak.idxmin()
            parent_color = df_color.iloc[index_min].parent
            child_color = df_color.iloc[index_min].child
            luminance = df_color.iloc[index_min].luminance
            brightness = df_color.iloc[index_min].brightness
            hue = df_color.iloc[index_min].hue
            brightness_lab = df_color.iloc[index_min].brightness_lab
            intensity =  df_color.iloc[index_min].intensity
            intensity_label =  df_color.iloc[index_min].intensity_label
            saturation =  df_color.iloc[index_min].saturation
            
            bucket.append({'file_name':file, 'parent_color':parent_color,'child_color':child_color, 'top':count,
                        'r':int(a[0]), 'g':int(a[1]), 'b':int(a[2]),
                          'h':int(h), 'l':int(l), 's':int(s),
                          'b_med':int(med), 'b_std':int(std),
                          'luminance':int(luminance),
                          'brightness':int(brightness),
                          'hue':int(hue),
                          'brightness_lab':brightness_lab,
                          'intensity':int(intensity),
                           'intensity_label':intensity_label,
                            'saturation':saturation})
            count += 1
    except:
        dominant_color = 999, 999, 999

# Generate Data Frame
df = pd.DataFrame(bucket)
df = df[columns_order]

df.to_csv('result.csv',sep=';')
print("Debug: done")