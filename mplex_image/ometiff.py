####
# title: mpimage.py
#
# language: Python3.6
# date: 2019-05-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 library to display, normalize and crop multiplex images
####

#libraries
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import pandas as pd
#import bioformats 
import re
import shutil
from itertools import chain
import matplotlib.ticker as ticker

os.chdir('/home/groups/graylab_share/OMERO.rdsStore/engje/Data/cmIF/')
from apeer_ometiff_library import omexmlClass

#functions

def gen_xml(array, channel_names):
    '''
    copy and modify from apeer ome tiff
    ls_marker
    '''
    #for idx, s_marker in enumerate(ls_marker):
    #    old = bytes(f'Name="C:{idx}"','utf-8')
    #    new = bytes(f'Name="{s_marker}"','utf-8')
    #    s_xml = s_xml.replace(old,new,-1)
    #Dimension order is assumed to be TZCYX
    dim_order = "TZCYX"
    
    metadata = omexmlClass.OMEXML()
    shape = array.shape
    assert ( len(shape) == 5), "Expected array of 5 dimensions"
    
    metadata.image().set_Name("IMAGE")
    metadata.image().set_ID("0")
    
    pixels = metadata.image().Pixels
    pixels.ome_uuid = metadata.uuidStr
    pixels.set_ID("0")
    
    pixels.channel_count = shape[2]
    
    pixels.set_SizeT(shape[0])
    pixels.set_SizeZ(shape[1])
    pixels.set_SizeC(shape[2])
    pixels.set_SizeY(shape[3])
    pixels.set_SizeX(shape[4])
    
    pixels.set_DimensionOrder(dim_order[::-1])
    
    pixels.set_PixelType(omexmlClass.get_pixel_type(array.dtype))
    
    for i in range(pixels.SizeC):
        pixels.Channel(i).set_ID("Channel:0:" + str(i))
        pixels.Channel(i).set_Name(channel_names[i])
    
    for i in range(pixels.SizeC):
        pixels.Channel(i).set_SamplesPerPixel(1)
        
    pixels.populate_TiffData()
    
    return metadata.to_xml().encode()
