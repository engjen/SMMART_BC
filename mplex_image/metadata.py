####
# title: metadata.py
#
# language: Python3.7
# date: 2020-07-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 library using python bioformats to extract image metadata
####


#libraries
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import pandas as pd
import bioformats 
#import javabridge
import re
import shutil
from itertools import chain, compress
import matplotlib.ticker as ticker
from mplex_image import cmif

# mpimage
#functions

def get_exposure(s_image, s_find="Information\|Image\|Channel\|ExposureTime\<\/Key\>\<Value\>"):

    s_meta = bioformats.get_omexml_metadata(path=s_image)
    o = bioformats.OMEXML(s_meta)
    print(o.image().Name)
    print(o.image().AcquisitionDate)

    li_start = [m.start() for m in re.finditer(s_find, s_meta)]
    if len(li_start)!=1:
        print('Error: found wrong number of exposure times')

    ls_exposure = []
    for i_start in li_start:
        ls_exposure.append(s_meta[i_start:i_start+200])
    s_exposure =  ls_exposure[0].strip(s_find)
    s_exposure = s_exposure[1:s_exposure.find(']')]
    ls_exposure = s_exposure.split(',')
    li_exposure = [int(item)/1000000 for item in ls_exposure]
    return(li_exposure,s_meta)

def get_exposure_sample(s_sample,df_img):
    """
    return a dataframe with all exposure times for a sample (slide)
    """
    #make dataframe of exposure time metadata
    df_exposure = pd.DataFrame()
    ls_image = os.listdir()
    df_sample = df_img[df_img.index.str.contains(s_sample)]
    for s_image in df_sample.index:
                        print(s_image)
                        li_exposure, s_meta = get_exposure(s_image)
                        se_times = pd.Series(li_exposure,name=s_image)
                        df_exposure = df_exposure.append(se_times)
    return(df_exposure)

def get_meta(s_image, s_find = 'Scene\|CenterPosition\<\/Key\>\<Value\>\['):
    """czi scene metadata
    s_image = filename
    s_find = string to find in the omexml metadata
    returns: 
    ls_exposure = list of 200 character strings following s_find in metadata
    s_meta = the whole metadata string
    """
    s_meta = bioformats.get_omexml_metadata(path=s_image)
    o = bioformats.OMEXML(s_meta)
    #print(o.image().Name)
    #print(o.image().AcquisitionDate)

    li_start = [m.start() for m in re.finditer(s_find, s_meta)]
    if len(li_start)!=1:
        print('Error: found wrong number of exposure times')

    ls_exposure = []
    for i_start in li_start:
        ls_exposure.append(s_meta[i_start:i_start+200])
    s_exposure =  ls_exposure[0].strip(s_find)
    s_exposure = s_exposure[0:s_exposure.find(']')]
    ls_exposure = s_exposure.split(',')
    #li_exposure = [int(item)/1000000 for item in ls_exposure]
    return(ls_exposure,s_meta)

def scene_position(czidir,type):
    """
    get a dataframe of scene positions for each round/scene in TMA
    """
    os.chdir(f'{czidir}')
    df_img = cmif.parse_czi('.',type=type)

    #javabridge.start_vm(class_path=bioformats.JARS)
    for s_image in df_img.index:
        print(s_image)
        ls_exposure,s_meta = get_meta(s_image)
        df_img.loc[s_image,'Scene_X'] = ls_exposure[0]
        df_img.loc[s_image,'Scene_Y'] = ls_exposure[1]

    #javabridge.kill_vm()

    df_img = df_img.sort_values(['rounds','scanID','scene']).drop('data',axis=1)
    return(df_img)


    ls_exposure,s_meta = get_meta(s_image, s_find = 'Scene\|CenterPosition\<\/Key\>\<Value\>\[')

def exposure_times_scenes(df_img,codedir,czidir,s_end='.czi'):
    """
    get a csv of exposure times for each slide
    """
    #go to directory
    os.chdir(czidir)
    #export exposure time
    s_test = sorted(compress(os.listdir(),[item.find(s_end) > -1 for item in os.listdir()]))[1]#[0]
    s_find = f"{s_test.split('-Scene-')[1].split(s_end)[0]}"
    for s_sample in sorted(set(df_img.slide)):
        print(s_sample)
        df_img_slide = df_img[(df_img.slide==s_sample) & (df_img.scene==s_find)]
        print(len(df_img_slide))
        df_exp = get_exposure_sample(s_sample,df_img_slide)
        df_exp.to_csv(f'{codedir}/{s_sample}_ExposureTimes.csv',header=True,index=True)

def exposure_times(df_img,codedir,czidir):
    """
    get a csv of exposure times for each slide
    """
    #go to directory
    os.chdir(czidir)
    print(czidir)
    #export exposure time
    for s_sample in sorted(set(df_img.slide)):
        df_img_slide = df_img[df_img.slide==s_sample]
        df_exp = get_exposure_sample(s_sample,df_img_slide)
        df_exp.to_csv(f'{codedir}/{s_sample}_ExposureTimes.csv',header=True,index=True)
    #close java virtual machine
    #javabridge.kill_vm()

def exposure_times_slide(df_img,codedir,czidir):
    if len(df_img.scene.unique()) == 1:
        exposure_times(df_img,codedir,czidir)
    elif len(df_img.scene.unique()) > 1:
        exposure_times_scenes(df_img,codedir,czidir,s_end='.czi')

def export_tiffs(df_img, s_sample,tiffdir):
    """
    export the tiffs of each tile
    """
    #start java virtual machine
    #javabridge.start_vm(class_path=bioformats.JARS)

    #export tiffs
    df_img_slide = df_img[df_img.slide==s_sample]
    for path in df_img_slide.index:
        print(path)
        img = bioformats.load_image(path) #looks like it only loads the first tile
        img_new = img*65535
        img_16 = img_new.astype(np.uint16)
        i_channels = img_16.shape[2]
        for i_channel in range(i_channels):
           print(f'channel {i_channel}')
           bioformats.write_image(f'{tiffdir}/{path.split(".czi")[0]}_c{str(i_channel+1)}_ORG.tif', pixels=img_16[:,:,i_channel],pixel_type='uint16')
           break
        break
    a_test = img_16[:,:,i_channel]
    aa_test = img_16
    #javabridge.kill_vm()
    return(a_test,aa_test, img)
