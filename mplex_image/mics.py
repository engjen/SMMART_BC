# wrapper functions for codex image processing

from mplex_image import preprocess, mpimage, getdata, process, analyze, cmif, features, ometiff
import os
import pandas as pd
import math
import skimage
from skimage import io, filters
import re
import numpy as np
import json
from skimage.util import img_as_uint
import tifffile

def parse_processed():
    '''
    parse the file names of processed Macsima images
    '''
    df_img = mpimage.filename_dataframe(s_end ="ome.tif",s_start='R',s_split='___')
    #standardize dapi naming
    ls_dapi_index = df_img[df_img.index.str.contains('DAPI')].index.tolist()
    d_replace = dict(zip(ls_dapi_index, [item.replace('DAPIV0','DAPI__DAPIV0') for item in ls_dapi_index]))
    df_img['data'] = df_img.data.replace(d_replace)
    #standardize AF naming
    ls_dapi_index = df_img[df_img.index.str.contains('autofluorescence')].index.tolist()
    d_replace = dict(zip(ls_dapi_index, [item.replace('autofluorescence_FITC','autofluorescence-FITC__FITC') for item in ls_dapi_index]))
    df_img['data'] = df_img.data.replace(d_replace)
    d_replace = dict(zip(ls_dapi_index, [item.replace('autofluorescence_PE','autofluorescence-PE__PE') for item in ls_dapi_index]))
    df_img['data'] = df_img.data.replace(d_replace)
    #standardize empty naming
    ls_dapi_index = df_img[df_img.index.str.contains('empty')].index.tolist()
    d_replace = dict(zip(ls_dapi_index, [item.replace('empty','empty__empty') for item in ls_dapi_index]))
    df_img['data'] = df_img.data.replace(d_replace)
    df_img['marker'] = [item.split(f"{item.split('_')[3]}_")[-1].split('__')[0] for item in df_img.data]
    df_img['cycle'] = [item.split('_')[3] for item in df_img.data]
    df_img['rounds'] = [item.split('_')[3].replace('C-','R') for item in df_img.data]
    df_img['clone'] = [item.split('__')[1].split('.')[0] for item in df_img.data]
    #standardize marker naming
    d_replace = dict(zip(df_img.marker.tolist(),[item.replace('_','-') for item in df_img.marker.tolist()]))
    df_img['data'] =  [item.replace(f'''{item.split(f"{item.split('_')[3]}_")[-1].split('__')[0]}''',f'''{d_replace[item.split(f"{item.split('_')[3]}_")[-1].split('__')[0]]}''') for item in df_img.data]
    df_img['exposure'] = [int(item.split('__')[1].split('_')[1].split('.')[0]) for item in df_img.data]
    df_img['channel'] = [item.split('__')[1].split('_')[0].split('.')[1] for item in df_img.data]
    d_replace = {'DAPI':'c1', 'FITC':'c2', 'PE':'c3', 'APC':'c4'}
    df_img['color'] = [item.replace(item, d_replace[item]) for item in df_img.channel]
    df_img['rack'] = [item.split('_')[0] for item in df_img.data]
    df_img['slide'] = [item.split('_')[1] for item in df_img.data]
    df_img['scene'] = [item.split('_')[2] for item in df_img.data]
    return(df_img)

def parse_org():
    '''
    parse the file names of copied (name-stadardized) Macsima images
    '''
    s_path = os.getcwd()
    df_img = mpimage.filename_dataframe(s_end ="tif",s_start='R',s_split='___')
    df_img['marker'] = [item.split(f"{item.split('_')[3]}_")[-1].split('__')[0] for item in df_img.data]
    df_img['cycle'] = [item.split('_')[3] for item in df_img.data]
    df_img['rounds'] = [item.split('_')[3].replace('C-','R') for item in df_img.data]
    df_img['clone'] = [item.split('__')[1].split('.')[0] for item in df_img.data]
    df_img['exposure'] = [int(item.split('__')[1].split('_')[1].split('.')[0]) for item in df_img.data]
    df_img['channel'] = [item.split('__')[1].split('_')[0].split('.')[1] for item in df_img.data]
    d_replace = {'DAPI':'c1', 'FITC':'c2', 'PE':'c3', 'APC':'c4'}
    df_img['color'] = [item.replace(item, d_replace[item]) for item in df_img.channel]
    df_img['rack'] = [item.split('_')[0] for item in df_img.data]
    df_img['slide'] = [item.split('_')[1] for item in df_img.data]
    df_img['scene'] = [item.split('_')[2] for item in df_img.data]
    df_img['slide_scene'] = df_img.slide + '_' + df_img.scene
    df_img['path'] = [f"{s_path}/{item}" for item in df_img.index]
    return(df_img)

def copy_processed(df_img,regdir,i_lines=32639):
    '''
    copy the highest exposure time images for processing
    '''
    for s_marker in sorted(set(df_img.marker) - {'DAPI','autofluorescence','empty'}):
        df_marker = df_img[df_img.marker==s_marker]
        for s_cycle in sorted(set(df_marker.cycle)):
            for s_index in df_marker[df_marker.cycle==s_cycle].sort_values('exposure',ascending=False).index.tolist():
                a_img = io.imread(s_index)
                s_dir_new = s_index.split(f"_{df_img.loc[s_index,'cycle']}")[0]
                s_index_new = df_img.loc[s_index,'data'].split('.ome.tif')[0]
                preprocess.cmif_mkdir([f'{regdir}/{s_dir_new}'])
                print(a_img.max())
                #get rid of lines
                a_img[a_img==i_lines] = a_img.min()
                if a_img.max() < 65535:
                    io.imsave(f'{regdir}/{s_dir_new}/{s_index_new}.tif',a_img,plugin='tifffile',check_contrast=False)
                    break
                else:
                    print('Try lower exposure time')
    for s_index in df_img[df_img.marker=='DAPI'].index.tolist():
        a_img = io.imread(s_index)
        print(f'DAPI max: {a_img.max()}')
        if df_img.loc[s_index,'rounds'] != 'R0': #keep lines in R0 dapi, for segmentation
            a_img[a_img==i_lines] = a_img.min()
        s_dir_new = s_index.split(f"_{df_img.loc[s_index,'cycle']}")[0]
        s_index_new = df_img.loc[s_index,'data'].split('.ome.tif')[0]
        preprocess.cmif_mkdir([f'{regdir}/{s_dir_new}'])
        io.imsave(f'{regdir}/{s_dir_new}/{s_index_new}.tif',a_img,plugin='tifffile',check_contrast=False)

def extract_cellpose_features(s_sample, segdir, regdir, ls_seg_markers, nuc_diam, cell_diam):
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
        if s_file.find(f'{".".join(ls_seg_markers)} nuc{nuc_diam} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split(f'_{".".join(ls_seg_markers)}')[0])
            d_match.update({s_file.split(f'_{".".join(ls_seg_markers)}')[0]:s_file})
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
        df_feat = features.extract_feat(labels,dapi, properties=(['label']))
        df_feat.columns = [f'{item}_segmented-nuclei' for item in df_feat.columns]
        df_feat.index = [f'{s_sample}_cell{item}' for item in df_feat.loc[:,'label_segmented-nuclei']]

        #get subcellular regions
        cyto = features.label_difference(labels,cell_labels)
        d_loc_nuc = features.subcellular_regions(labels, distance_short=2, distance_long=5)
        d_loc_cell = features.subcellular_regions(cell_labels, distance_short=2, distance_long=5)
        d_loc = {'nuclei':labels,'cell':cell_labels,'cytoplasm':cyto,
         'nucmem':d_loc_nuc['membrane'][0],'cellmem':d_loc_cell['membrane'][0],
         'perinuc5':d_loc_nuc['ring'][1],'exp5':d_loc_nuc['grown'][1],
         'nucadj2':d_loc_nuc['straddle'][0],'celladj2':d_loc_cell['straddle'][0]}

        #subdir organized by slide or scene
        if os.path.exists(f'{regdir}/{s_sample}'):
            os.chdir(f'{regdir}/{s_sample}')
        elif os.path.exists(f'{regdir}/{s_scene}'):
            os.chdir(f'{regdir}/{s_scene}')
        else:
            os.chdir(f'{regdir}')
        df_img = parse_org()
        df_img['round_int'] = [int(re.sub('[^0-9]','', item)) for item in df_img.rounds] 
        df_img = df_img[df_img.round_int < 90]
        df_img = df_img.sort_values('round_int')
        #take into account slide (well)
        df_scene = df_img[df_img.slide_scene==s_scene]
        #load each image
        for s_index in df_scene.index:
                intensity_image = io.imread(s_index)
                df_thresh.loc[s_index,'threshold_li'] =  filters.threshold_li(intensity_image)
                if intensity_image.mean() > 0:
                    df_thresh.loc[s_index,'threshold_otsu'] = filters.threshold_otsu(intensity_image)
                    df_thresh.loc[s_index,'threshold_triangle'] = filters.threshold_triangle(intensity_image)
                s_marker = df_scene.loc[s_index,'marker']
                print(f'extracting features {s_marker}')
                if s_marker == 'DAPI':
                    s_marker = s_marker + f'{df_scene.loc[s_index,"rounds"].split("R")[1]}'
                for s_loc, a_loc in d_loc.items():
                    if s_loc == 'nuclei':
                        df_marker_loc = features.extract_feat(a_loc,intensity_image, properties=(['mean_intensity','centroid','area','eccentricity','label']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_centroid-0',f'{s_marker}_{s_loc}_centroid-1',f'{s_marker}_{s_loc}_area',f'{s_marker}_{s_loc}_eccentricity',f'{s_marker}_{s_loc}_label']
                    elif s_loc == 'cell':
                        df_marker_loc = features.extract_feat(a_loc,intensity_image, properties=(['mean_intensity','euler_number','area','eccentricity','label']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_euler',f'{s_marker}_{s_loc}_area',f'{s_marker}_{s_loc}_eccentricity',f'{s_marker}_{s_loc}_label']
                    else:
                        df_marker_loc = features.extract_feat(a_loc,intensity_image, properties=(['mean_intensity','label']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_label']
                    #set array ids as index
                    df_marker_loc.index = df_marker_loc.loc[:,f'{s_marker}_{s_loc}_label']
                    df_marker_loc.index = [f'{s_sample}_cell{item}' for item in df_marker_loc.index]
                    df_feat = df_feat.merge(df_marker_loc, left_index=True,right_index=True,how='left',suffixes=('',f'{s_marker}_{s_loc}'))
        df_sample = df_sample.append(df_feat)
    return(df_sample, df_thresh)

def combine_labels(s_sample,segdir, subdir, ls_seg_markers, nuc_diam, cell_diam, df_mi_full,s_thresh):
    '''
    - load cell labels; delete cells that were not used for cytoplasm (i.e. ecad neg)
    - nuc labels, expand to perinuc 5 and then cut out the cell labels
    - keep track of cells that are completely coverd by another cell (or two or three: counts as touching).
    '''
    se_neg = df_mi_full[df_mi_full.slide == s_sample].loc[:,f'{s_thresh}_negative']
    dd_result = {}
    if os.path.exists(f'{segdir}/{s_sample}Cellpose_Segmentation'):
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
    else:
        os.chdir(segdir)
    ls_scene = []
    for s_file in os.listdir():
        if s_file.find(' - DAPI.png') > -1:
            ls_scene.append(s_file.split(' - DAPI.png')[0])
    ls_scene = sorted(set(df_mi_full[df_mi_full.slide == s_sample].scene) & set(ls_scene))
    for s_scene in ls_scene:
        se_neg_scene = se_neg[se_neg.index.str.contains(s_scene)]

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
        perinuc5, labels_exp = features.expand_label(labels,distance=distance)
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

def cropped_ometiff(s_sample,subdir,cropdir,d_crop,d_combos,s_dapi,tu_dim):
    if os.path.exists(f'{subdir}/{s_sample}'):
        os.chdir(f'{subdir}/{s_sample}')
    df_img = parse_org()
    df_img['scene'] = s_sample
    d_crop_scene = {s_sample:d_crop[s_sample]}
    dd_result = mpimage.overlay_crop(d_combos,d_crop_scene,df_img,s_dapi,tu_dim)
    for s_crop, d_result in dd_result.items():
        for s_type, (ls_marker, array) in d_result.items():
            print(f'Generating multi-page ome-tiff {[item for item in ls_marker]}')
            new_array = array[np.newaxis,np.newaxis,:]
            s_xml =  ometiff.gen_xml(new_array, ls_marker)
            with tifffile.TiffWriter(f'{cropdir}/{s_crop}_{s_type}.ome.tif') as tif:
                tif.save(new_array,  photometric = "minisblack", description=s_xml, metadata = None)


#old
def convert_dapi(debugdir,regdir,b_mkdir=True):
    '''
    convert dapi to tif, rename to match Guillaumes pipeline requirements
    '''
    cwd = os.getcwd()
    os.chdir(debugdir)
    for s_dir in sorted(os.listdir()):
        if s_dir.find('R-1_')== 0:
            os.chdir(s_dir)
            for s_file in sorted(os.listdir()):
                if s_file.find('bleach')==-1:
                    s_round = s_file.split("Cycle(")[1].split(").ome.tif")[0]
                    print(f'stain {s_round}')
                    s_dir_new = s_dir.split('_')[2] + '-Scene-0' + s_dir.split('F-')[1]
                    s_tissue_dir = s_dir.split('_F-')[0]
                    if b_mkdir:
                        preprocess.cmif_mkdir([f'{regdir}/{s_tissue_dir}'])
                    a_dapi = skimage.io.imread(s_file)
                    #rename with standard name (no stain !!!!)
                    with skimage.external.tifffile.TiffWriter(f'{regdir}/{s_tissue_dir}/{s_dir_new}_R{s_round}_DAPI_V0_c1_ORG_5.0.tif') as tif:
                        tif.save(a_dapi)
            os.chdir('..')
    os.chdir(cwd)

def convert_channels(processdir, regdir, b_rename=True, testbool=True):
    '''
    convert channels to tif, select one exposure time of three, rename to match Guillaumes pipeline requirements
    '''
    cwd = os.getcwd()
    os.chdir(processdir)
    for s_dir in sorted(os.listdir()):
        if s_dir.find('R-1_')== 0:
            os.chdir(s_dir)
            if b_rename:
                d_rename = {'autofluorescencePE_P':'autofluorescencePE_V0_P',
                'autofluorescenceFITC_F':'autofluorescenceFITC_V0_F',
                '000_DAPIi':'extra000_DAPIi',
                '000_DAPIf':'extra000_DAPIf',
                'extraextraextra':'extra',
                'extraextra':'extra',
                '_FITC_':'_c2_ORG_',
                '_PE_':'_c3_ORG_',}
                preprocess.dchange_fname(d_rename,b_test=testbool)
                
            #parse file names
            else:
                ls_column = ['rounds','marker','dilution','fluor','ORG','exposure','expdecimal','imagetype1','imagetype']
                df_img = mpimage.parse_img(s_end =".tif",s_start='0',s_sep1='_',s_sep2='.',ls_column=ls_column,b_test=False)
                df_img['exposure'] = df_img.exposure.astype(dtype='int')
                ls_marker = sorted(set(df_img.marker))
                for s_marker in ls_marker:
                    df_marker = df_img[df_img.marker==s_marker]
                    df_sort = df_marker.sort_values(by=['exposure'],ascending=False,inplace=False)
                    for idx in range(len(df_sort.index)):
                        s_index = df_sort.index[idx]
                        a_img = skimage.io.imread(s_index)
                        df_file = df_sort.loc[s_index,:]
                        print(a_img.max())
                        if idx < len(df_sort.index) - 1:
                            if a_img.max() < 65535:
                                print(f'Selected {df_file.exposure} for {df_file.marker}')
                                s_dir_new = s_dir.split('_')[2] + '-Scene-0' + s_dir.split('F-')[1]
                                s_tissue_dir = s_dir.split('_F-')[0]
                                s_index_new = s_index.split(".ome.tif")[0]
                                with skimage.external.tifffile.TiffWriter(f'{regdir}/{s_tissue_dir}/{s_dir_new}_R{s_index_new}.tif') as tif:
                                    tif.save(a_img)
                                break
                            else:
                                print('Try lower exposure time')
                        elif idx == len(df_sort.index) - 1:
                            print(f'Selected as the lowest exposure time {df_file.exposure} for {df_file.marker}')
                            s_dir_new = s_dir.split('_')[2] + '-Scene-0' + s_dir.split('F-')[1]
                            s_tissue_dir = s_dir.split('_F-')[0]
                            s_index_new = s_index.split(".ome.tif")[0]
                            with skimage.external.tifffile.TiffWriter(f'{regdir}/{s_tissue_dir}/{s_dir_new}_R{s_index_new}.tif') as tif:
                                tif.save(a_img)
                        else:
                            print('/n /n /n /n Error in finding exposure time')
        
            os.chdir('..')

def parse_converted(regdir):
        '''
        parse the converted miltenyi file names,
        regdir contains the images
        '''
        s_dir = os.getcwd()
        df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='G',s_split='_')
        df_img.rename({'data':'scene'},axis=1,inplace=True)
        df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
        df_img['marker'] = [item[2] for item in [item.split('_') for item in df_img.index]]
        df_img['dilution'] = [item[3] for item in [item.split('_') for item in df_img.index]]
        df_img['color'] = [item[4] for item in [item.split('_') for item in df_img.index]]
        df_img['scene_int'] = [item.split('Scene-')[1] for item in df_img.scene]
        df_img['scene_int'] = df_img.scene_int.astype(dtype='int')
        df_img['exposure'] = [item[6].split('.')[0] for item in [item.split('_') for item in df_img.index]]
        df_img['path'] = [f'{regdir}/{s_dir}/{item}' for item in df_img.index]
        df_img['tissue'] = s_dir
        return(df_img)

def parse_converted_dirs(regdir):
    '''
    parse the converted miltenyi file names,
    regdir is the master folder containing subfolders with ROIs/gates
    '''
    os.chdir(regdir)
    df_img_all = pd.DataFrame()
    for idx, s_dir in enumerate(sorted(os.listdir())):
        os.chdir(s_dir)
        s_sample = s_dir
        print(s_sample)
        df_img = parse_converted(s_dir)
        df_img_all = df_img_all.append(df_img)
        os.chdir('..')
    return(df_img_all)

def count_images(df_img,b_tile_count=True):
    """
    count and list slides, scenes, rounds
    """
    df_count = pd.DataFrame(index=sorted(set(df_img.scene)),columns=sorted(set(df_img.color)))
    for s_sample in sorted(set(df_img.tissue)):
        print(f'ROI {s_sample}')
        df_img_slide = df_img[df_img.tissue==s_sample]
        print('tiles')
        [print(item) for item in sorted(set(df_img_slide.scene))]
        print(f'Number of images = {len(df_img_slide)}')
        print(f'Rounds:')
        [print(item) for item in sorted(set(df_img_slide.rounds))]
        print('\n')
        if b_tile_count:
            for s_scene in sorted(set(df_img_slide.scene)):
                df_img_scene = df_img_slide[df_img_slide.scene==s_scene]
                for s_color in sorted(set(df_img_scene.color)):
                    print(f'{s_scene} {s_color} {len(df_img_scene[df_img_scene.color==s_color])}')
                    df_count.loc[s_scene,s_color] = len(df_img_scene[df_img_scene.color==s_color])
    return(df_count)

def visualize_reg_images(regdir,qcdir,color='c1',tu_array=(3,2)):
    """
    array registered images to check tissue identity, focus, etc.
    """
    #check registration
    preprocess.cmif_mkdir([f'{qcdir}/RegisteredImages'])
    cwd = os.getcwd()
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        os.chdir(s_dir)
        s_sample = s_dir
        print(s_sample)
        df_img = parse_converted(s_dir)
        ls_scene = sorted(set(df_img.scene))
        for s_scene in ls_scene:
            print(s_scene)
            df_img_scene = df_img[df_img.scene == s_scene]
            df_img_stain = df_img_scene[df_img_scene.color==color]
            df_img_sort = df_img_stain.sort_values(['rounds'])
            i_sqrt = math.ceil(math.sqrt(len(df_img_sort)))
            #array_img(df_img,s_xlabel='color',ls_ylabel=['rounds','exposure'],s_title='marker',tu_array=(2,4),tu_fig=(10,20))
            if color == 'c1':
                fig = mpimage.array_img(df_img_sort,s_xlabel='marker',ls_ylabel=['rounds','exposure'],s_title='rounds',tu_array=tu_array,tu_fig=(16,14))
            else:
                fig = mpimage.array_img(df_img_sort,s_xlabel='color',ls_ylabel=['rounds','exposure'],s_title='marker',tu_array=tu_array,tu_fig=(16,12))
            fig.savefig(f'{qcdir}/RegisteredImages/{s_scene}_registered_{color}.png')
        os.chdir('..')
    os.chdir(cwd)
    #return(df_img)

def rename_files(d_rename,dir,b_test=True):
    """
    change file names
    """
    cwd = os.getcwd()
    os.chdir(dir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{dir}/{s_dir}'
        os.chdir(s_path)
        print(s_dir)
        df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='reg',s_split='_')
        df_img.rename({'data':'scene'},axis=1,inplace=True)
        df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
        df_img['color'] = [item[2] for item in [item.split('_') for item in df_img.index]]
        df_img['marker'] = [item[3].split('.')[0] for item in [item.split('_') for item in df_img.index]]
        if b_test:
            print('This is a test')
            preprocess.dchange_fname(d_rename,b_test=True)
        elif b_test==False:
            print('Changing name - not a test')
            preprocess.dchange_fname(d_rename,b_test=False)
        else:
            pass

def rename_fileorder(s_sample, dir, b_test=True):
    """
    change file names
    """
    cwd = os.getcwd()
    os.chdir(dir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{dir}/{s_dir}'
        os.chdir(s_path)
        print(s_dir)
        df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='Scene',s_split='_')
        df_img.rename({'data':'scene'},axis=1,inplace=True)
        df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
        df_img['color'] = [item[2] for item in [item.split('_') for item in df_img.index]]
        df_img['marker'] = [item[3].split('.')[0] for item in [item.split('_') for item in df_img.index]]
        for s_index in df_img.index:
            s_round = df_img.loc[s_index,'rounds']
            s_scene= f"{s_sample}-{df_img.loc[s_index,'scene']}"
            s_marker = df_img.loc[s_index,'marker']
            s_color = df_img.loc[s_index,'color']
            s_index_rename = f'{s_round}_{s_scene}_{s_marker}_{s_color}_ORG.tif'
            d_rename = {s_index:s_index_rename}
            if b_test:
                print('This is a test')
                preprocess.dchange_fname(d_rename,b_test=True)
            elif b_test==False:
                print('Changing name - not a test')
                preprocess.dchange_fname(d_rename,b_test=False)
            else:
                pass


def copy_files(dir,dapi_copy, marker_copy,testbool=True,type='codex'):
    """
    copy and rename files if needed as dummies
    need to edit
    """
    os.chdir(dir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{dir}/{s_dir}'
        os.chdir(s_path)
        #s_sample = s_dir.split('-Scene')[0]
        df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='Scene',s_split='_')
        df_img.rename({'data':'scene'},axis=1,inplace=True)
        df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
        df_img['color'] = [item[2] for item in [item.split('_') for item in df_img.index]]
        df_img['marker'] = [item[3].split('.')[0] for item in [item.split('_') for item in df_img.index]]
        print(s_dir)
        #if b_test:
        for key, dapi_item in dapi_copy.items():
                df_dapi = df_img[(df_img.rounds== key.split('_')[1]) & (df_img.color=='c1')]
                s_dapi = df_dapi.loc[:,'marker'][0]
                preprocess.copy_dapis(s_r_old=key,s_r_new=f'_cyc{dapi_item}_',s_c_old='_c1_',
                 s_c_new='_c2_',s_find=f'_c1_{s_dapi}_ORG.tif',b_test=testbool,type=type)
        i_count=0
        for idx,(key, item) in enumerate(marker_copy.items()):
                preprocess.copy_markers(df_img, s_original=key, ls_copy = item,
                 i_last_round= dapi_item + i_count, b_test=testbool,type=type)
                i_count=i_count + len(item)

def segmentation_thresholds(regdir,qcdir, d_segment):
    """
    visualize binary mask of segmentaiton threholds
    need to edit
    """
    preprocess.cmif_mkdir([f'{qcdir}/Segmentation'])
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='Scene',s_split='_')
        df_img.rename({'data':'scene'},axis=1,inplace=True)
        df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
        df_img['color'] = [item[2] for item in [item.split('_') for item in df_img.index]]
        df_img['marker'] = [item[3].split('.')[0] for item in [item.split('_') for item in df_img.index]]
        s_sample = s_dir
        print(s_sample)
        d_seg = preprocess.check_seg_markers(df_img,d_segment, i_rows=1, t_figsize=(6,6)) #few scenes
        for key, fig in d_seg.items():
            fig.savefig(f'{qcdir}/Segmentation/{s_dir}_{key}_segmentation.png')


def segmentation_inputs(s_sample,regdir,segdir,d_segment,b_start=False):
    """
    make inputs for guillaumes segmentation
    """
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='R',s_split='_')
        df_img.rename({'data':'rounds'},axis=1,inplace=True)
        #df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
        df_img['color'] = [item[3] for item in [item.split('_') for item in df_img.index]]
        df_img['marker'] = [item[2] for item in [item.split('_') for item in df_img.index]]
        #s_sample = s_dir
        #s_sample = s_dir.split('-Scene')[0]
        print(s_sample)
        df_marker = df_img[df_img.color!='c1']
        df_marker = df_marker.sort_values(['rounds','color'])
        df_dapi = pd.DataFrame(index = [df_marker.marker.tolist()],columns=['rounds','colors','minimum','maximum','exposure','refexp','location'])
        df_dapi['rounds'] = df_marker.loc[:,['rounds']].values
        df_dapi['colors'] = df_marker.loc[:,['color']].values
        df_dapi['minimum'] = 1003
        df_dapi['maximum'] = 65535
        df_dapi['exposure'] = 100
        df_dapi['refexp'] = 100
        df_dapi['location'] = 'All'
        for s_key,i_item in d_segment.items():
            df_dapi.loc[s_key,'minimum'] = i_item
        df_dapi.to_csv('RoundsCyclesTable.txt',sep=' ',header=False)
        df_dapi.to_csv(f'metadata_{s_sample}_RoundsCyclesTable.csv',header=True)
        #create cluster.java file
        preprocess.cluster_java(s_dir=f'JE{idx}',s_sample=s_sample,imagedir=f'{s_path}',segmentdir=segdir,type='exacloud',b_segment=True,b_TMA=False)
        if b_start:
            os.chdir(f'/home/groups/graylab_share/Chin_Lab/ChinData/Work/engje/exacloud/JE{idx}') #exacloud
            print(f'JE{idx}')
            os.system('make_sh')
