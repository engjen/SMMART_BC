####
# title: features.py
# language: Python3.7
# date: 2020-06-00
# license: GPL>=v3
# author: Jenny
# description:
#   python3 script for single cell feature extraction
####

#libraries
import os
import sys
import numpy as np
import pandas as pd
import shutil
import skimage
import scipy
from scipy import stats
from scipy import ndimage as ndi
from skimage import measure, segmentation, morphology
from skimage import io, filters
import re
import json
from biotransistor import imagine
from PIL import Image
from mplex_image import process
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = 1000000000

#functions
def extract_feat(labels,intensity_image, properties=('centroid','mean_intensity','area','eccentricity')):
    ''' 
    given labels and intensity image, extract features to dataframe
    '''
    props = measure.regionprops_table(labels,intensity_image, properties=properties)
    df_prop = pd.DataFrame(props)
    return(df_prop)

def expand_label(labels,distance=3):
    '''
    expand the nucelar labels by a fixed number of pixels
    '''
    boundaries = segmentation.find_boundaries(labels,mode='outer') #thick
    shrunk_labels = labels.copy()
    shrunk_labels[boundaries] = 0
    background = shrunk_labels == 0
    distances, (i, j) = scipy.ndimage.distance_transform_edt(
                background, return_indices=True
            )

    grown_labels = labels.copy()
    mask = background & (distances <= distance)
    grown_labels[mask] = shrunk_labels[i[mask], j[mask]]
    ring_labels = grown_labels - shrunk_labels

    return(ring_labels, grown_labels) #shrunk_labels, grown_labels,

def contract_label(labels,distance=3):
    '''
    contract labels by a fixed number of pixels
    '''
    boundaries = segmentation.find_boundaries(labels,mode='outer')
    shrunk_labels = labels.copy()
    shrunk_labels[boundaries] = 0
    foreground = shrunk_labels != 0
    distances, (i, j) = scipy.ndimage.distance_transform_edt(
                     foreground, return_indices=True
                 )

    mask = foreground & (distances <= distance)
    shrunk_labels[mask] = shrunk_labels[i[mask], j[mask]]
    rim_labels = labels - shrunk_labels
    return(rim_labels)

def straddle_label(labels,distance=3):
    '''
    expand and contract labels by a fixed number of pixels
    '''
    boundaries = segmentation.find_boundaries(labels,mode='outer') #outer
    shrunk_labels = labels.copy()
    grown_labels = labels.copy()
    shrunk_labels[boundaries] = 0
    foreground = shrunk_labels != 0
    background = shrunk_labels == 0
    distances_f, (i, j) = scipy.ndimage.distance_transform_edt(
                     foreground, return_indices=True
                 )
    distances_b, (i, j) = scipy.ndimage.distance_transform_edt(
                background, return_indices=True
            )
    mask_f = foreground & (distances_f <= distance)
    mask_b = background & (distances_b <= distance + 1)
    shrunk_labels[mask_f] = 0
    grown_labels[mask_b] = grown_labels[i[mask_b], j[mask_b]]
    membrane_labels = grown_labels - shrunk_labels 
    return(membrane_labels, grown_labels, shrunk_labels)

def label_difference(labels,cell_labels):
    '''
    given matched nuclear and cell label IDs,return cell_labels minus labels
    '''
    overlap = cell_labels==labels
    ring_rep = cell_labels.copy()
    ring_rep[overlap] = 0
    return(ring_rep)

def get_mip(ls_img):
    '''
    maximum intensity projection of images (input list of filenames)
    '''
    imgs = []
    for s_img in ls_img:
        img = io.imread(s_img)
        imgs.append(img)
    mip = np.stack(imgs).max(axis=0)
    return(mip)

def thresh_li(img,area_threshold=100,low_thresh=1000):
    '''
    threshold an image with Li’s iterative Minimum Cross Entropy method
    if too low, apply the low threshold instead (in case negative)
    '''
    mask = img >= filters.threshold_li(img)
    mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)
    mask[mask < low_thresh] = 0
    return(mask)

def mask_border(mask,type='inner',pixel_distance = 50):
    '''
    for inner, distance transform from mask to background
    for outer, distance transform from back ground to mask
    returns a mask
    '''
    shrunk_mask = mask.copy()
    if type == 'inner':
        foreground = ~mask
        background = mask
    elif type == 'outer':
        foreground = ~mask
        background = mask
    distances, (i, j) = scipy.ndimage.distance_transform_edt(
                background, return_indices=True
            )
    maskdist = mask & (distances <= pixel_distance)
    shrunk_mask[maskdist] = shrunk_mask[i[maskdist], j[maskdist]]
    mask_out = np.logical_and(mask,np.logical_not(shrunk_mask))
    return(mask_out,shrunk_mask,maskdist,distances)

def mask_labels(mask,labels):
    ''''
    return the labels that fall within the mask
    '''
    selected_array = labels[mask]
    a_unique = np.unique(selected_array)
    return(a_unique)

def parse_org(s_end = "ORG.tif",s_start='R'):
    """
    This function will parse images following koei's naming convention
    Example: Registered-R1_PCNA.CD8.PD1.CK19_Her2B-K157-Scene-002_c1_ORG.tif
    The output is a dataframe with image filename in index
    And rounds, color, imagetype, scene (/tissue), and marker in the columns
    """
    ls_file = []
    for file in os.listdir():
        if file.endswith(s_end):
            if file.find(s_start)==0:
                ls_file = ls_file + [file]
    df_img = pd.DataFrame(index=ls_file)
    df_img['rounds'] = [item.split('_')[0].split('Registered-')[1] for item in df_img.index]
    df_img['color'] = [item.split('_')[-2] for item in df_img.index]
    df_img['slide'] = [item.split('_')[2] for item in df_img.index]
    df_img['scene'] = [item.split('-Scene-')[1] for item in df_img.slide]
    #parse file name for biomarker
    for s_index in df_img.index:
        #print(s_index)
        s_color = df_img.loc[s_index,'color']
        if s_color == 'c1':
            s_marker = 'DAPI'
        elif s_color == 'c2':
            s_marker = s_index.split('_')[1].split('.')[0]
        elif s_color == 'c3':
            s_marker = s_index.split('_')[1].split('.')[1]
        elif s_color == 'c4':
            s_marker = s_index.split('_')[1].split('.')[2]
        elif s_color == 'c5':
            s_marker = s_index.split('_')[1].split('.')[3]
        elif s_color == 'c6':
            s_marker = s_index.split('_')[1].split('.')[2]
        elif s_color == 'c7':
            s_marker = s_index.split('_')[1].split('.')[3]
        else: print('Error')
        df_img.loc[s_index,'marker'] = s_marker
    return(df_img) 

def extract_cellpose_features(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam,b_big=False): #,b_thresh=False
    '''
    load the segmentation results, the input images, and the channels images
    extract mean intensity from each image, and centroid, area and eccentricity for 
    '''

    df_sample = pd.DataFrame()
    df_thresh = pd.DataFrame()

    os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
    ls_scene = []
    d_match = {}
    for s_file in os.listdir():
        if s_file.find(f'{".".join(ls_seg_markers)} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split('_')[0])
            d_match.update({s_file.split('_')[0]:s_file})
        elif s_file.find(f'{".".join(ls_seg_markers)} nuc{nuc_diam} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split('_')[0])
            d_match.update({s_file.split('_')[0]:s_file})
    for s_scene in ls_scene:
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
        print(f'processing {s_scene}')
        for s_file in os.listdir():
            if s_file.find(s_scene) > -1:
                if s_file.find("DAPI.png") > -1:
                    s_dapi = s_file
        dapi = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{s_dapi}')
        print(f'loading {s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        labels = io.imread(f'{s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        cell_labels = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{d_match[s_scene]}')
        print(f'loading {d_match[s_scene]}')
        #nuclear features
        df_feat = extract_feat(labels,dapi, properties=(['label']))
        df_feat.columns = [f'{item}_segmented-nuclei' for item in df_feat.columns]
        df_feat.index = [f'{s_sample}_scene{s_scene.split("-Scene-")[1].split("_")[0]}_cell{item}' for item in df_feat.loc[:,'label_segmented-nuclei']]

        #get subcellular regions
        cyto = label_difference(labels,cell_labels)
        d_loc_nuc = subcellular_regions(labels, distance_short=2, distance_long=5)
        d_loc_cell = subcellular_regions(cell_labels, distance_short=2, distance_long=5)
        d_loc = {'nuclei':labels,'cell':cell_labels,'cytoplasm':cyto,
         'nucmem':d_loc_nuc['membrane'][0],'cellmem':d_loc_cell['membrane'][0],
         'perinuc5':d_loc_nuc['ring'][1],'exp5':d_loc_nuc['grown'][1],
         'nucadj2':d_loc_nuc['straddle'][0],'celladj2':d_loc_cell['straddle'][0]}

        #subdir organized by slide or scene
        if os.path.exists(f'{subdir}/{s_sample}'):
            os.chdir(f'{subdir}/{s_sample}')
        elif os.path.exists(f'{subdir}/{s_scene}'):
            os.chdir(f'{subdir}/{s_scene}')
        else:
            os.chdir(f'{subdir}')
        df_img = parse_org()
        df_img['round_int'] = [int(re.sub('[^0-9]','', item)) for item in df_img.rounds] 
        df_img = df_img[df_img.round_int < 90]
        df_img = df_img.sort_values('round_int')
        df_scene = df_img[df_img.scene==s_scene.split("-Scene-")[1].split("_")[0]]

        #load each image
        for s_index in df_scene.index:
                intensity_image = io.imread(s_index)
                df_thresh.loc[s_index,'threshold_li'] =  filters.threshold_li(intensity_image)
                if intensity_image.mean() > 0:
                    df_thresh.loc[s_index,'threshold_otsu'] = filters.threshold_otsu(intensity_image)
                    df_thresh.loc[s_index,'threshold_triangle'] = filters.threshold_triangle(intensity_image)
                #if b_thresh:
                #    break
                s_marker = df_scene.loc[s_index,'marker']
                print(f'extracting features {s_marker}')
                if s_marker == 'DAPI':
                    s_marker = s_marker + f'{df_scene.loc[s_index,"rounds"].split("R")[1]}'
                for s_loc, a_loc in d_loc.items():
                    if s_loc == 'nuclei':
                        df_marker_loc = extract_feat(a_loc,intensity_image, properties=(['mean_intensity','centroid','area','eccentricity','label']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_centroid-0',f'{s_marker}_{s_loc}_centroid-1',f'{s_marker}_{s_loc}_area',f'{s_marker}_{s_loc}_eccentricity',f'{s_marker}_{s_loc}_label']
                    elif s_loc == 'cell':
                        df_marker_loc = extract_feat(a_loc,intensity_image, properties=(['mean_intensity','euler_number','area','eccentricity','label']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_euler',f'{s_marker}_{s_loc}_area',f'{s_marker}_{s_loc}_eccentricity',f'{s_marker}_{s_loc}_label']
                    else:
                        df_marker_loc = extract_feat(a_loc,intensity_image, properties=(['mean_intensity','label']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_label']
                    #drop zero from array, set array ids as index
                    #old df_marker_loc.index = sorted(np.unique(a_loc)[1::])
                    df_marker_loc.index = df_marker_loc.loc[:,f'{s_marker}_{s_loc}_label']
                    df_marker_loc.index = [f'{s_sample}_scene{s_scene.split("-Scene-")[1].split("_")[0]}_cell{item}' for item in df_marker_loc.index]
                    df_feat = df_feat.merge(df_marker_loc, left_index=True,right_index=True,how='left',suffixes=('',f'{s_marker}_{s_loc}'))
        if b_big:
            df_feat.to_csv(f'{segdir}/{s_sample}Cellpose_Segmentation/features_{s_sample}-{s_scene}.csv')
        df_sample = df_sample.append(df_feat)
    return(df_sample, df_thresh)

def extract_bright_features(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam,ls_membrane):
    '''
    load the features, segmentation results, the input images, and the channels images
    extract mean intensity of the top 25% of pixel in from each label region
    '''
    df_sample = pd.DataFrame()
    os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
    ls_scene = []
    d_match = {}
    for s_file in os.listdir():
        if s_file.find(f'{".".join(ls_seg_markers)} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split('_')[0])
            d_match.update({s_file.split('_')[0]:s_file})
        elif s_file.find(f'{".".join(ls_seg_markers)} nuc{nuc_diam} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split('_')[0])
            d_match.update({s_file.split('_')[0]:s_file})
    for s_scene in ls_scene:
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
        print(f'processing {s_scene}')
        for s_file in os.listdir():
            if s_file.find(s_scene) > -1:
                if s_file.find("DAPI.png") > -1:
                    s_dapi = s_file
        dapi = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{s_dapi}')
        print(f'loading {s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        labels = io.imread(f'{s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        print(labels.shape)
        cell_labels = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{d_match[s_scene]}')
        print(cell_labels.shape)
        print(f'loading {d_match[s_scene]}')
        #nuclear features
        df_feat = extract_feat(labels,dapi, properties=(['label']))
        df_feat.columns = [f'{item}_segmented-nuclei' for item in df_feat.columns]
        df_feat.index = [f'{s_sample}_scene{s_scene.split("-Scene-")[1].split("_")[0]}_cell{item}' for item in df_feat.loc[:,'label_segmented-nuclei']]

        #get subcellular regions
        d_loc_nuc = subcellular_regions(labels, distance_short=2, distance_long=5)
        d_loc_cell = subcellular_regions(cell_labels, distance_short=2, distance_long=5)
        d_loc = {'nucmem25':d_loc_nuc['membrane'][0],'exp5nucmembrane25':d_loc_nuc['grown'][1],
            'cellmem25':d_loc_cell['membrane'][0],'nuclei25':labels}

        #subdir organized by slide or scene
        if os.path.exists(f'{subdir}/{s_sample}'):
            os.chdir(f'{subdir}/{s_sample}')
        elif os.path.exists(f'{subdir}/{s_scene}'):
            os.chdir(f'{subdir}/{s_scene}')
        else:
            os.chdir(f'{subdir}')
        df_img = parse_org()
        df_img['round_int'] = [int(re.sub('[^0-9]','', item)) for item in df_img.rounds] 
        df_img = df_img[df_img.round_int < 90]
        df_img = df_img.sort_values('round_int')
        df_scene = df_img[df_img.scene==s_scene.split("-Scene-")[1].split("_")[0]]
        df_marker = df_scene[df_scene.marker.isin(ls_membrane)]
        #load each image
        for s_index in df_marker.index:
                print(f'loading {s_index}')
                intensity_image = io.imread(s_index)
                #print(intensity_image.shape)
                s_marker = df_marker.loc[s_index,'marker']
                print(f'extracting features {s_marker}')
                if s_marker == 'DAPI':
                    s_marker = s_marker + f'{df_marker.loc[s_index,"rounds"].split("R")[1]}'
                for s_loc, a_loc in d_loc.items():
                    #print(a_loc.shape)
                    df_marker_loc = pd.DataFrame(columns = [f'{s_marker}_{s_loc}'])
                    df_prop = extract_feat(a_loc,intensity_image, properties=(['intensity_image','image','label']))
                    for idx in df_prop.index:
                        label_id = df_prop.loc[idx,'label']
                        intensity_image_small = df_prop.loc[idx,'intensity_image']
                        image = df_prop.loc[idx,'image']
                        pixels = intensity_image_small[image]
                        pixels25 = pixels[pixels >= np.quantile(pixels,.75)]
                        df_marker_loc.loc[label_id,f'{s_marker}_{s_loc}'] = pixels25.mean()
                    df_marker_loc.index = [f'{s_sample}_scene{s_scene.split("-Scene-")[1].split("_")[0]}_cell{item}' for item in df_marker_loc.index]
                    df_feat = df_feat.merge(df_marker_loc, left_index=True,right_index=True,how='left',suffixes=('',f'{s_marker}_{s_loc}'))
        df_sample = df_sample.append(df_feat)
        #break
    return(df_sample)

def subcellular_regions(labels, distance_short=2, distance_long=5):
    '''
    calculate subcellular segmentation regions from segmentation mask
    '''
    membrane_short = contract_label(labels,distance=distance_short)
    membrane_long = contract_label(labels,distance=distance_long)
    ring_short, grown_short = expand_label(labels,distance=distance_short)
    ring_long, grown_long = expand_label(labels,distance=distance_long)
    straddle_short, __, shrink_short = straddle_label(labels,distance=distance_short)
    straddle_long, __, shrink_long = straddle_label(labels,distance=distance_long)
    d_loc_sl={'membrane':(membrane_short,membrane_long),
     'ring':(ring_short,ring_long),
     'straddle':(straddle_short,straddle_long),
     'grown':(grown_short,grown_long),
     'shrunk':(shrink_short,shrink_long)}
    return(d_loc_sl)
 
def combine_labels(s_sample,segdir, subdir, ls_seg_markers, nuc_diam, cell_diam, df_mi_full,s_thresh):
    '''
    - load cell labels; delete cells that were not used for cytoplasm (i.e. ecad neg)
    - nuc labels, expand to perinuc 5 and then cut out the cell labels
    - keep track of cells that are completely coverd by another cell (or two or three: counts as touching).
    '''
    se_neg = df_mi_full[df_mi_full.slide == s_sample].loc[:,f'{s_thresh}_negative']
    print(len(se_neg))
    dd_result = {}
    if os.path.exists(f'{segdir}/{s_sample}Cellpose_Segmentation'):
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
    else:
        os.chdir(segdir)
        print(segdir)
    ls_scene = []
    for s_file in os.listdir():
        if s_file.find(' - DAPI.png') > -1:
            ls_scene.append(s_file.split(' - DAPI.png')[0])
    ls_scene_all = sorted(set([item.split('_cell')[0].replace('_scene','-Scene-') for item in se_neg.index]) & set(ls_scene))
    if len(ls_scene_all) == 0:
        ls_scene_all = sorted(set([item.split('_cell')[0].replace('_scene','-Scene-').split('_')[1] for item in se_neg.index]) & set(ls_scene))
    print(ls_scene_all)
    for s_scene in ls_scene_all:
        se_neg_scene = se_neg[se_neg.index.str.contains(s_scene.replace("Scene ","scene")) | se_neg.index.str.contains(s_scene.replace("-Scene-","_scene"))]
        print(f'Processing combined segmentaiton labels for {s_scene}')
        if os.path.exists(f'{s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif'):
            labels = io.imread(f'{s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        else:
            print('no nuclei labels found')
        if os.path.exists(f'{s_scene} matchedcell{cell_diam} - Cell Segmentation Basins.tif'):
            cell_labels = io.imread(f'{s_scene} matchedcell{cell_diam} - Cell Segmentation Basins.tif')
        elif os.path.exists(f'{s_scene}_{".".join(ls_seg_markers)} matchedcell{cell_diam} - Cell Segmentation Basins.tif'):
            cell_labels = io.imread(f'{s_scene}_{".".join(ls_seg_markers)} matchedcell{cell_diam} - Cell Segmentation Basins.tif')
        elif os.path.exists(f'{s_scene}_{".".join(ls_seg_markers)} nuc{nuc_diam} matchedcell{cell_diam} - Cell Segmentation Basins.tif'):
            cell_labels = io.imread(f'{s_scene}_{".".join(ls_seg_markers)} nuc{nuc_diam} matchedcell{cell_diam} - Cell Segmentation Basins.tif')
        else:
            print('no cell labels found')
        #set non-ecad cell labels to zero
        a_zeros = np.array([int(item.split('_cell')[1]) for item in se_neg_scene[se_neg_scene].index]).astype('int64')
        mask = np.isin(cell_labels, a_zeros)
        cell_labels_copy = cell_labels.copy()
        cell_labels_copy[mask] = 0
        #make the nuclei under cells zero
        labels_copy = labels.copy()
        distance = 5
        perinuc5, labels_exp = expand_label(labels,distance=distance)
        labels_exp[cell_labels_copy > 0] = 0
        #combine calls and expanded nuclei
        combine = (labels_exp + cell_labels_copy)
        if s_scene.find('Scene') == 0:
            io.imsave(f'{s_sample}_{s_scene.replace("Scene ","scene")}_cell{cell_diam}_nuc{nuc_diam}_CombinedSegmentationBasins.tif',combine)
        else:
            io.imsave(f'{s_scene}_{".".join(ls_seg_markers)}-cell{cell_diam}_exp{distance}_CellSegmentationBasins.tif',combine)
        #figure out the covered cells...labels + combined
        not_zero_pixels =  np.array([labels.ravel() !=0,combine.ravel() !=0]).all(axis=0)
        a_tups = np.array([combine.ravel()[not_zero_pixels],labels.ravel()[not_zero_pixels]]).T #combined over nuclei
        unique_rows = np.unique(a_tups, axis=0)
        new_dict = {}
        for key, value in unique_rows:
            if key == value:
                continue
            else:
                if key in new_dict:
                    new_dict[key].append(value)
                else:
                    new_dict[key] = [value]
        #from elmar (reformat cells touching dictionary and save
        d_result = {}
        for i_cell, li_touch in new_dict.items():
            d_result.update({str(i_cell): [str(i_touch) for i_touch in li_touch]})
        dd_result.update({f'{s_sample}_{s_scene.replace("Scene ","scene")}':d_result})
    #save dd_touch as json file
    with open(f'result_{s_sample}_cellsatop_dictionary.json','w') as f: 
        json.dump(dd_result, f)
    print('')
    return(labels,combine,dd_result)

def check_basins(cell_labels, cell_diam):
    dai_value = {'a':cell_labels}
    df = imagine.membrane_px(cell_labels,dai_value)
    ls_bad = sorted(set(df[df.x_relative > 10*cell_diam].cell) | set(df[df.y_relative > 10*cell_diam].cell))
    return(ls_bad)

def check_combined(segdir,s_sample,cell_diam,ls_seg_markers):
    df_result = pd.DataFrame()
    if os.path.exists(f'{segdir}/{s_sample}Cellpose_Segmentation'):
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
    else:
        os.chdir(segdir)
    ls_scene = []
    for s_file in os.listdir():
        if s_file.find(' - DAPI.png') > -1:
            ls_scene.append(s_file.split(' - DAPI.png')[0])
    for s_scene in sorted(ls_scene):
        print(s_scene)
        if os.path.exists(f'{s_scene}_{".".join(ls_seg_markers)}-cell{cell_diam}_exp5_CellSegmentationBasins.tif'):
            cell_labels = io.imread(f'{s_scene}_{".".join(ls_seg_markers)}-cell{cell_diam}_exp5_CellSegmentationBasins.tif')
            print(f'Loaded {s_scene}_{".".join(ls_seg_markers)}-cell{cell_diam}_exp5_CellSegmentationBasins.tif')
            ls_bad = check_basins(cell_labels, cell_diam)
            ls_bad_cells = [f"{s_scene.replace('-Scene-','_scene')}_cell{item}" for item in ls_bad]
            df_bad = pd.DataFrame(index=ls_bad_cells,columns=['bad_match'],data=[True]*len(ls_bad_cells))
            df_result = df_result.append(df_bad)
        else:
            print('no combined cell labels found')
    return(df_result)

def edge_mask(s_sample,segdir,subdir,i_pixel=154, dapi_thresh=350,i_fill=50000,i_close=20):
    '''
    find edge of the tissue. first, find tissue by threshodling DAPI R1 (pixels above dapi_thresh)
    then, mask all pixels within i_pixel distance of tissue border
    return/save binary mask
    '''
    os.chdir(segdir)
    df_img = process.load_li([s_sample],s_thresh='', man_thresh=100)
    for s_scene in sorted(set(df_img.scene)):
        print(f'Calculating tissue edge mask for Scene {s_scene}')
        s_index = df_img[(df_img.scene == s_scene) & (df_img.rounds == 'R1') & (df_img.color =='c1')].index[0]
        if os.path.exists(f'{subdir}/{s_sample}/{s_index}'):
            img_dapi = io.imread(f'{subdir}/{s_sample}/{s_index}')
        elif os.path.exists(f'{subdir}/{s_sample}-Scene-{s_scene}/{s_index}'):
            img_dapi = io.imread(f'{subdir}/{s_sample}-Scene-{s_scene}/{s_index}')
        else:
            print('no DAPI found')
            img_dapi = np.zeros([2,2])
        mask = img_dapi > dapi_thresh 
        mask_small = morphology.remove_small_objects(mask, min_size=100)
        mask_closed = morphology.binary_closing(mask_small, morphology.octagon(i_close,i_close//2))
        mask_filled = morphology.remove_small_holes(mask_closed, i_fill)
        border_mask, __, __,distances = mask_border(mask_filled,type='inner',pixel_distance = i_pixel)
        img = np.zeros(border_mask.shape,dtype='uint8')
        img[border_mask] = 255
        io.imsave(f"{segdir}/TissueEdgeMask{i_pixel}_{s_sample}_scene{s_scene}.png", img)

def edge_hull(s_sample,segdir,subdir,i_pixel=154, dapi_thresh=350,i_fill=50000,i_close=40,i_small=30000):
    '''
    find edge of the tissue. first, find tissue by threshodling DAPI R1 (pixels above dapi_thresh)
    then, mask all pixels within i_pixel distance of tissue border
    return/save binary mask
    '''
    os.chdir(segdir)
    df_img = process.load_li([s_sample],s_thresh='', man_thresh=100)
    for s_scene in sorted(set(df_img.scene)):
        print(f'Calculating tissue edge mask for Scene {s_scene}')
        s_index = df_img[(df_img.scene == s_scene) & (df_img.rounds == 'R1') & (df_img.color =='c1')].index[0]
        if os.path.exists(f'{subdir}/{s_sample}/{s_index}'):
            img_dapi = io.imread(f'{subdir}/{s_sample}/{s_index}')
        elif os.path.exists(f'{subdir}/{s_sample}-Scene-{s_scene}/{s_index}'):
            img_dapi = io.imread(f'{subdir}/{s_sample}-Scene-{s_scene}/{s_index}')
        else:
            print('no DAPI found')
            img_dapi = np.zeros([2,2])
        mask = img_dapi > dapi_thresh 
        mask_small = morphology.remove_small_objects(mask, min_size=100)
        mask_closed = morphology.binary_closing(mask_small, morphology.octagon(i_close,i_close//2))
        mask_filled = morphology.remove_small_holes(mask_closed, i_fill)
        mask_smaller = morphology.remove_small_objects(mask, min_size=i_small)
        mask_hull = morphology.convex_hull_image(mask_smaller)
        border_mask, __, __,distances = mask_border(mask_filled,type='inner',pixel_distance = i_pixel)
        img = np.zeros(border_mask.shape,dtype='uint8')
        img[border_mask] = 255
        io.imsave(f"{segdir}/TissueEdgeMask{i_pixel}_{s_sample}_scene{s_scene}.png", img)

def edge_cells(s_sample,segdir,nuc_diam,i_pixel=154):
    '''
    load a binary mask of tissue, cell labels, and xy coord datafreame.
    return data frame of cells witin binary mask
    '''
    df_sample = pd.DataFrame()
    #load xy
    df_xy = pd.read_csv(f'{segdir}/features_{s_sample}_CentroidXY.csv',index_col=0)
    df_xy['cells'] = [int(item.split('cell')[1]) for item in df_xy.index]
    ls_scene = sorted(set([item.split('_')[1].split('scene')[1] for item in df_xy.index]))
    #load masks
    for s_scene in ls_scene:
        print(f'Calculating edge cells for Scene {s_scene}')
        mask = io.imread(f"{segdir}/TissueEdgeMask{i_pixel}_{s_sample}_scene{s_scene}.png")
        mask_gray = mask == 255
        labels = io.imread(f'{segdir}/{s_sample}Cellpose_Segmentation/{s_sample}-Scene-{s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        edge = mask_labels(mask_gray,labels)
        df_scene = df_xy[df_xy.index.str.contains(f'{s_sample}_scene{s_scene}')]
        #works
        es_cells = set(edge.astype('int')).intersection(set(df_scene.cells))
        df_edge = df_scene[df_scene.cells.isin(es_cells)]
        fig,ax=plt.subplots()
        ax.imshow(mask_gray)
        ax.scatter(df_edge.DAPI_X,df_edge.DAPI_Y,s=1)
        fig.savefig(f'{segdir}/TissueEdgeMask{i_pixel}_{s_sample}-Scene-{s_scene}_cells.png')
        df_sample = df_sample.append(df_edge)
    return(df_sample)

def cell_distances(df_xy,s_scene,distances):
    '''
    load a binary mask of tissue, cell labels, and xy coord datafreame.
    return data frame of cells witin binary mask
    '''
    df_xy['DAPI_Y'] = df_xy.DAPI_Y.astype('int64')
    df_xy['DAPI_X'] = df_xy.DAPI_X.astype('int64')
    print(f'Calculating distances for Scene {s_scene}')
    df_scene = df_xy[df_xy.index.str.contains(f"{s_scene.replace('-Scene-','_scene')}")].copy()
    df_scene['pixel_dist'] = distances[df_scene.DAPI_Y,df_scene.DAPI_X]
    return(df_scene)

def cell_coords():
    '''
    TBD: find cell coordinate within a mask
    '''
    for s_scene in ls_scene:
        #old (use if you have coordinates, not labels)
        #mask_gray = mask#[:,:,0]
        #contour = skimage.measure.find_contours(mask_gray,0)
        #coords = skimage.measure.approximate_polygon(contour[0], tolerance=5)
        #fig,ax=plt.subplots()
        #ax.imshow(mask_gray)
        #ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
        #fig.savefig(f'TissueEdgeMask_{s_sample}_Scene-{s_scene}_polygon.png')
        #x = np.array(df_scene.DAPI_X.astype('int').values)
        #y = np.array(df_scene.DAPI_Y.astype('int').values)
        #points = np.array((y,x)).T
        mask = skimage.measure.points_in_poly(points, coords)