####
# title: analyze.py
#
# language: Python3.6
# date: 2019-05-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 library to analyze cyclic data and images after manual thresholding
####

#load libraries
import matplotlib as mpl
mpl.use('agg')
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import json
from biotransistor import imagine
import itertools

#functions
# import importlib
# importlib.reload(analyze)

def combinations(df_tn_tumor,ls_marker=['CK19_Ring','CK7_Ring','CK5_Ring','CK14_Ring','CD44_Ring','Vim_Ring']):
    '''
    get all combinations of the markers (can be overlapping)
    '''
    ls_combos = []
    for i in range(0,len(ls_marker)):
        for tu_combo in itertools.combinations(ls_marker,i+1):#'Ecad_Ring',
            ls_combos.append(tu_combo)

    #create the combos dataframe dataframe
    df_tn_counts = pd.DataFrame(index=df_tn_tumor.index) 
    se_all = set(ls_marker)

    #combos of 2 or more
    for tu_combo in ls_combos:
        print(tu_combo)
        se_pos = df_tn_tumor[(df_tn_tumor.loc[:,tu_combo].sum(axis=1) ==len(tu_combo))] #those are pos
        se_neg = df_tn_tumor[(df_tn_tumor.loc[:,(se_all)].sum(axis=1) == len(tu_combo))] #and only those
        df_tn_counts['_'.join([item for item in tu_combo])] = df_tn_tumor.index.isin(se_pos.index.intersection(se_neg.index))
    
    #other cells (negative for all)
    df_tn_counts['__'] = df_tn_counts.loc[:,df_tn_counts.dtypes=='bool'].sum(axis=1)==0
    if sum(df_tn_counts.sum(axis=1)!=1) !=0:
        print('error in analyze.combinations')

    return(df_tn_counts)

def gated_combinations(df_data,ls_gate,ls_marker):
    '''
    df_data = boolean cell type dataframe
    ls_gate = combine each of these cell types (full coverage and non-overlapping)
    ls_marker = with these cell tpyes (full coverage and non-overlapping)
    '''
    es_all = set(ls_marker + ls_gate)
    ls_old = df_data.columns
    df_gate_counts = pd.DataFrame()
    for s_gate in ls_gate:
        df_tn_tumor = df_data[df_data.loc[:,s_gate]]
        print(f'{s_gate} {len(df_tn_tumor)}')
        #combos of 2
        if len(df_tn_tumor) >=1:
            for s_marker in ls_marker:
                print(s_marker)
                tu_combo = (s_gate,s_marker)
                es_neg = es_all - set(tu_combo)
                if ~df_data.loc[:,tu_combo].all(axis=1).any():
                    df_gate_counts[f"{s_gate}_{s_marker}"] = False
                else:
                    df_gate_counts[f"{s_gate}_{s_marker}"] = df_data.loc[:,tu_combo].all(axis=1) & ~df_data.loc[:,es_neg].any(axis=1)
    df_gate_counts.fillna(value=False, inplace=True)
    return(df_gate_counts) 

def add_celltype(df_data, ls_cell_names, s_type_name):
    '''
    add gated cell type to data frame, and save the possible cell typesand cell type name in a csv
    df_data = data frame with the cell types (boolean)
    ls_cell_names = list of the cell names
    s_type_name = the cell category
    '''
    #check cell types' exclusivity
    if ((df_data.loc[:,ls_cell_names].sum(axis=1)>1)).sum()!=0:
        print(f'Error in exclusive cell types: {s_type_name}')

    #make cell type object columns
    for s_marker in ls_cell_names:
        df_data.loc[(df_data[df_data.loc[:,s_marker]]).index,s_type_name] = s_marker
    d_record = {s_type_name:ls_cell_names}

    #append the record json
    if not os.path.exists('./Gating_Record.json'):
        with open(f'Gating_Record.json','w') as f: 
            json.dump(d_record, f, indent=4, sort_keys=True)
    else:
        with open('Gating_Record.json','r') as f:
            d_current = json.load(f)
        d_current.update(d_record)
        with open(f'Gating_Record.json','w') as f: 
            json.dump(d_current, f, indent=4, sort_keys=True)

def thresh_meanint(df_thresh,d_crop={},s_thresh='minimum',):
    """
    threshold, and output positive and negative mean intensity and array
    df_thresh = dataframe of images with columns having image attributes
        and index with image names, column with threshold values
    d_crop = image scene and crop coordinates

    """
    d_mask = {}
    for idx, s_index in enumerate(df_thresh.index):
        #load image, crop, thresh
        a_image = skimage.io.imread(s_index)
        if len(d_crop) != 0:
            tu_crop = d_crop[df_thresh.loc[s_index,'scene']]
            a_image = a_image[(tu_crop[1]):(tu_crop[1]+tu_crop[3]),(tu_crop[0]):(tu_crop[0]+tu_crop[2])]
        i_min = df_thresh.loc[s_index,s_thresh]
        a_mask = a_image > i_min
        print(f'mean positive intensity = {np.mean(a_image[a_mask])}')
        df_thresh.loc[s_index,'meanpos'] = np.mean(a_image[a_mask])
        b_mask = a_image < i_min
        print(f'mean negative intensity = {np.mean(a_image[b_mask])}')
        df_thresh.loc[s_index,'meanneg'] = np.mean(a_image[b_mask])
        d_mask.update({s_index:a_mask})
    return(df_thresh,d_mask)

def mask_meanint(df_img, a_mask):
    '''
    for each image in dataframe of image (df_img)
    calculate mean intensity in pixels in mask (a_mask)
    '''

    #for each image, calculate mean intensity in the masked area
    for s_index in df_img.index:
        a_img = skimage.io.imread(s_index)
        a_img_total = a_img[a_mask]
        i_img_meanint = a_img_total.sum()/a_img_total.size
        df_img.loc[s_index,'result'] = i_img_meanint
    return(df_img)

def make_border(s_sample,df_pos,ls_color,segmentdir,savedir,b_images=True,s_find = 'Cell Segmentation Basins.tif',s_split='Scene '): 
    """
    load positive cells dataframe, and segmentation basins
    output the borders od positive cells and the cells touching dictionary
    """
    #load segmentation basins 
    #flattens ids into a set (stored in d_flatten)
    os.chdir(segmentdir)
    ls_file = os.listdir()
    ls_cellseg = []

    # list of Basin files
    for s_file in ls_file:
        if s_file.find(s_find)>-1:
            if s_file.find(s_sample)>-1:
                ls_cellseg.append(s_file)

    d_flatten = {}
    dd_touch = {}

    for s_file in ls_cellseg:
        s_scene_num = s_file.split(s_split)[1].split('_')[0].split(' ')[0]
        print(s_file)
        print(s_scene_num)
        a_img = io.imread(s_file)
        # get all cell ids that exist in the images
        es_cell = set(a_img.flatten())
        es_cell.remove(0)
        s_scene = f'scene{s_scene_num}'
        d_flatten.update({f'scene{s_scene_num}':es_cell})

        #get a cell touching dictionary (only do this one (faster))
        dd_touch.update({f'{s_sample}_{s_scene}':imagine.touching_cells(a_img, i_border_width=0)})

        #s_type = 'Manual' 
        if b_images:
            #save png of cell borders (single tiffs)
            for idx, s_color in enumerate(ls_color):
                print(f'Processing {s_color}')
                #positive cells = positive cells based on thresholds
                #dataframe of all the positive cells
                df_color_pos = df_pos[df_pos.loc[:,s_color]]
                ls_index = df_color_pos.index.tolist()
 
                if len(df_color_pos[(df_color_pos.scene==s_scene)])>=1:
                    ls_index = df_color_pos[(df_color_pos.scene==s_scene)].index.tolist()
                    es_cell_positive = set([int(s_index.split('cell')[-1]) for s_index in ls_index])

                    # erase all non positive basins
                    es_cell_negative = d_flatten[s_scene].difference(es_cell_positive)
                    a_pos = np.copy(a_img)
                    a_pos[np.isin(a_img, list(es_cell_negative))] = 0   # bue: this have to be a list, else it will not work!

                    # get cell border (a_pos_border)
                    a_pos_border = imagine.get_border(a_pos)  # border has value 1
                    a_pos_border = np.uint16(a_pos_border * 65000)  # border will have value 255
                    #filename hack
                    print('saving image')
                    io.imsave(f'{savedir}/Registered-R{idx+100}_{s_color.replace("_",".")}.border.border.border_{df_color_pos.index[0].split("_")[0]}-{s_scene.replace("scene","Scene-")}_c2_ORG.tif',a_pos_border)
                else:
                    print(len(df_color_pos[(df_color_pos.scene==s_scene)]))
    #from elmar (reformat cells touching dictionary and save

    ddes_image = {}
    for s_image, dei_image in dd_touch.items():
        des_cell = {}
        for i_cell, ei_touch in dei_image.items():
            des_cell.update({str(i_cell): [str(i_touch) for i_touch in sorted(ei_touch)]})
        ddes_image.update({s_image:des_cell})

    #save dd_touch as json file
    with open(f'result_{s_sample}_cellstouching_dictionary.json','w') as f: 
        json.dump(ddes_image, f)
    return(ddes_image)

def make_border_all(s_sample,df_pos,segmentdir,savedir,b_images=True):
    """
    load positive cells dataframe, and segmentation basins
    output the borders od positive cells and the cells touching dictionary
    """
    #Specify which images to save
    #ls_color = df_pos.columns.tolist()
    #ls_color.remove('DAPI_X')
    #ls_color.remove('DAPI_Y')
    #ls_color.remove('scene')

    #load segmentation basins 
    #flattens ids into a set (stored in d_flatten)
    os.chdir(segmentdir)
    ls_file = os.listdir()
    ls_cellseg = []
    d_files = {}
    #dictionary of file to scene ID , and a list of Basin files
    for s_file in ls_file:
        if s_file.find('Cell Segmentation Basins.tif')>-1:
            if s_file.find(s_sample)>-1:
                ls_cellseg.append(s_file)
                s_scene_num = s_file.split(' ')[1]
                d_files.update({f'scene{s_scene_num}':s_file})

    d_flatten = {}
    dd_touch = {}

    for s_file in ls_cellseg:
        s_scene_num = s_file.split(' ')[1]
        print(s_file)
        a_img = skimage.io.imread(s_file)
        # get all cell ids that exist in the images
        es_cell = set(a_img.flatten())
        es_cell.remove(0)
        s_scene = f'scene{s_scene_num}'
        d_flatten.update({f'scene{s_scene_num}':es_cell})

        #get a cell touching dictionary (only do this one (faster))
        dd_touch.update({f'{s_sample}_{s_scene}':imagine.touching_cells(a_img, i_border_width=0)})

        #s_type = 'Manual' 
        if b_images:
                idx=0
            #save png of all cell borders (single tiffs)
            #for idx, s_color in enumerate(ls_color):
            #    print(f'Processing {s_color}')
                #positive cells = positive cells based on thresholds
                #dataframe of all the positive cells
                df_color_pos = df_pos #[df_pos.loc[:,s_color]]
                ls_index = df_color_pos.index.tolist()
 
                if len(df_color_pos[(df_color_pos.scene==s_scene)])>=1:
                    ls_index = df_color_pos[(df_color_pos.scene==s_scene)].index.tolist()
                    es_cell_positive = set([int(s_index.split('cell')[-1]) for s_index in ls_index])

                    # erase all non positive basins
                    es_cell_negative = d_flatten[s_scene].difference(es_cell_positive)
                    a_pos = np.copy(a_img)
                    a_pos[np.isin(a_img, list(es_cell_negative))] = 0   # bue: this have to be a list, else it will not work!

                    # get cell border (a_pos_border)
                    a_pos_border = imagine.get_border(a_pos)  # border has value 1
                    a_pos_border = a_pos_border.astype(np.uint8)
                    a_pos_border = a_pos_border * 255  # border will have value 255
                    #filename hack 2019-11-27
                    skimage.io.imsave(f'{savedir}/R{idx+100}_all.all_{df_color_pos.index[0].split("_")[0]}-{s_scene.replace("scene","Scene-")}_border_c3_ORG.tif',a_pos_border)

def celltype_to_bool(df_data, s_column):
    """
    Input a dataframe and column name of cell tpyes
    Output a new boolean dataframe with each col as a cell type
    """
    df_bool = pd.DataFrame(index=df_data.index)
    for celltype in sorted(set(df_data.loc[:,s_column])):
        df_bool.loc[df_data[df_data.loc[:,s_column]==celltype].index,celltype] = True
    df_bool = df_bool.fillna(value=False)
    df_data.columns = [str(item) for item in df_data.columns]
    return(df_bool)