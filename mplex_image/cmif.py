# wrapper functions for cmIF image processing

from mplex_image import preprocess, mpimage, getdata, process, features, register, ometiff
import copy
import time
import os
import numpy as np
import shutil
import subprocess
import pandas as pd
import math
from itertools import compress
import skimage
import sys
import re
from skimage import io
from skimage.util import img_as_uint
import tifffile

#set src path (CHANGE ME)
s_src_path = '/home/groups/graylab_share/OMERO.rdsStore/engje/Data/cmIF'
s_work_path = '/home/groups/graylab_share/Chin_Lab/ChinData/Work/engje'


def parse_czi(czidir,type='r',b_scenes=True):
    """
    parse .czi's written in koei's naming convention
    type = 's' for stitched
    """
    cwd = os.getcwd()
    #go to directory
    os.chdir(czidir)
    df_img = mpimage.filename_dataframe(s_end = ".czi",s_start='R',s_split='_')
    df_img['slide'] = [item[2] for item in [item.split('_') for item in df_img.index]]
    if type=='s':
        df_img['slide'] = [item[5] for item in [item.split('_') for item in df_img.index]]
    df_img['rounds'] = [item[0] for item in [item.split('_') for item in df_img.index]]
    df_img['markers'] = [item[1] for item in [item.split('_') for item in df_img.index]]
    if b_scenes:
        try:
            df_img['scene'] = [item[1].split('.')[0] for item in [item.split('Scene-') for item in df_img.index]]
        except IndexError:
            print(f"{set([item[0] for item in [item.split('Scene-') for item in df_img.index]])}")
        df_img['scanID'] = [item[-1].split('-Scene')[0] for item in [item.split('__') for item in df_img.index]]
    os.chdir(cwd)
    return(df_img)

def parse_stitched_czi(czidir,s_slide,b_scenes=True):
    '''
    parse .czi's wtitten in koei's naming convention, with periods changed to undescores
    '''
    cwd = os.getcwd()
    #go to directory
    os.chdir(czidir)
    df_img = mpimage.filename_dataframe(s_end = ".czi",s_start='R',s_split='_').rename({'data':'rounds'},axis=1)
    df_img['markers'] = [item[0] for item in [item.split(f'_{s_slide}') for item in df_img.index]]
    for s_index in df_img.index:
        df_img.loc[s_index,'markers_un'] = df_img.loc[s_index,'markers'].split(f"{df_img.loc[s_index,'rounds']}_")[1]
    df_img['markers'] = df_img.markers_un.str.replace('_','.')
    df_img.slide = s_slide
    if b_scenes:
        df_img['scene'] = [item[1].split('-')[0] for item in [item.split('Scene-') for item in df_img.index]]
    os.chdir(cwd)
    return(df_img)

def count_images(df_img):
    """
    count and list slides, scenes, rounds
    """
    for s_sample in sorted(set(df_img.slide)):
        print(s_sample)
        df_img_slide = df_img[df_img.slide==s_sample]
        print('scene names')
        [print(f'{item}: {sum(df_img_slide.scene==item)}') for item in sorted(set(df_img_slide.scene))]
        print(f'Number of images = {len(df_img_slide)}')
        print(f'Rounds:')
        [print(f'{item}: {sum(df_img_slide.rounds==item)}') for item in sorted(set(df_img_slide.rounds))]
        print('\n')

def visualize_raw_images(df_img,qcdir,color='c1'):
    """
    array raw images to check tissue identity, focus, etc.
    """
    for s_sample in sorted(set(df_img.slide)):
        print(s_sample)

        df_img_slide = df_img[df_img.slide==s_sample]
        for s_scene in sorted(set(df_img_slide.scene)):
            print(s_scene)
            df_dapi = df_img_slide[(df_img_slide.color==color) & (df_img_slide.scene==s_scene)].sort_values(['round_ord','rounds'])
            fig = mpimage.array_img(df_dapi,s_xlabel='slide',ls_ylabel=['scene','color'],s_title='rounds',tu_array=(2,len(df_dapi)//2+1),tu_fig=(24,10))
            fig.savefig(f'{qcdir}/RawImages/{s_sample}-Scene-{s_scene}_{color}_all.png')

def registration_python(s_sample,tiffdir,regdir,qcdir):
    print(f'Registering {s_sample}')
    preprocess.cmif_mkdir([f'{qcdir}/RegistrationPlots/'])
    os.chdir(f'{tiffdir}/{s_sample}')
    df_img = mpimage.parse_org(s_end = "ORG.tif",type='raw')
    df_img['round_ord'] = [int(re.sub('[^0-9]','', item)) for item in df_img.rounds] 
    df_img = df_img.sort_values(['round_ord','rounds','color','scene'])
    for i_scene in sorted(set(df_img.scene)):
        preprocess.cmif_mkdir([f'{regdir}/{s_sample}-Scene-{i_scene}'])
        df_dapi = df_img[(df_img.color=='c1') & (df_img.scene==i_scene)]
        target_file = df_dapi[df_dapi.rounds=='R1'].index[0]
        target = io.imread(target_file)
        for moving_file in df_dapi.index:
            s_round = moving_file.split('_')[0]
            moving_pts, target_pts, transformer = register.register(target_file,moving_file,b_plot=True)
            for moving_channel in df_img[(df_img.rounds==s_round) & (df_img.scene==i_scene)].index:
                moving = io.imread(moving_channel)
                warped_img, warped_pts = register.apply_transform(moving, target, moving_pts, target_pts, transformer)
                warped_img = img_as_uint(warped_img)
                io.imsave(f"{regdir}/{s_sample}-Scene-{i_scene}/Registered-{moving_channel.split(s_sample)[0]}{s_sample}-Scene-{moving_channel.split('-Scene-')[1]}",warped_img)

def run_registration_matlab(d_register, ls_order, tiffdir, regdir, N_colors='5'):
    """
    run registration on server with or without cropping
    """
    os.chdir(tiffdir)
    shutil.copyfile(f'{s_src_path}/src/wrapper.sh', './wrapper.sh')
    for s_sample, d_crop in d_register.items():
        if len(d_crop) > 0:
            print(f'Large registration {s_sample}')
            for key, value in d_crop.items():
                if len(str(key)) == 1:
                    preprocess.cmif_mkdir([f'{regdir}/{s_sample.split("-Scene")[0]}-Scene-00{str(key)}'])
                elif len(str(key)) == 2:
                    preprocess.cmif_mkdir([f'{regdir}/{s_sample.split("-Scene")[0]}-Scene-0{str(key)}'])
            preprocess.large_registration_matlab(N_smpl='10000',N_colors=N_colors,s_rootdir=tiffdir, s_subdirname=regdir,
             d_crop_regions=d_crop, s_ref_id='./R1_*_c1_ORG.tif', ls_order=ls_order) 
            MyOut = subprocess.Popen(['sbatch', 'wrapper.sh'], #the script runs fine
             stdout=subprocess.PIPE,
             stderr=subprocess.STDOUT)
        #regular registration
        else:
            print(f'Regular registration {s_sample}')
            df_img = mpimage.parse_org(s_end = "ORG.tif",type='raw')
            df_img['slide_scene'] = df_img.slide + '-Scene-' + df_img.scene
            preprocess.cmif_mkdir([(f'{regdir}/{item}') for item in sorted(set(df_img.slide_scene))]) #this will break with diff slides
            preprocess.registration_matlab(N_smpl='10000',N_colors=N_colors,s_rootdir=tiffdir, s_subdirname=f'{regdir}/',
             s_ref_id='./R1_*_c1_ORG.tif',ls_order =ls_order)
            MyOut = subprocess.Popen(['sbatch', 'wrapper.sh'], #the script runs fine
             stdout=subprocess.PIPE,
             stderr=subprocess.STDOUT)

def visualize_reg_images(regdir,qcdir,color='c1',s_sample=''):
    """
    array registered images to check tissue identity, focus, etc.
    """
    #check registration
    preprocess.cmif_mkdir([f'{qcdir}/RegisteredImages'])
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        if s_dir.find(s_sample) > -1:
            os.chdir(s_dir)
            s_sample_name = s_dir.split('-Scene')[0]
            print(s_sample_name)
            df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
            ls_scene = sorted(set(df_img.scene))
            for s_scene in ls_scene:
                print(s_scene)
                df_img_scene = df_img[df_img.scene == s_scene]
                df_img_stain = df_img_scene[df_img_scene.color==color]
                df_img_sort = df_img_stain.sort_values(['round_ord','rounds'])
                i_sqrt = math.ceil(math.sqrt(len(df_img_sort)))
                fig = mpimage.array_img(df_img_sort,s_xlabel='marker',ls_ylabel=['scene','color'],s_title='rounds',tu_array=(2,len(df_img_sort)//2+1),tu_fig=(24,10))
                #fig = mpimage.array_img(df_img_sort,s_column='color',s_row='rounds',s_label='scene',tu_array=(i_sqrt,i_sqrt),tu_fig=(16,14))
                fig.savefig(f'{qcdir}/RegisteredImages/{s_scene}_registered_{color}.png')
            os.chdir('..')
    return(df_img_sort)

def rename_files(d_rename,dir,b_test=True):
    """
    change file names
    """
    os.chdir(dir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{dir}/{s_dir}'
        os.chdir(s_path)
        #s_sample = s_dir.split('-Scene')[0]
        print(s_dir)
        df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
        es_wrong= preprocess.check_names(df_img)
        if b_test:
            print('This is a test')
            preprocess.dchange_fname(d_rename,b_test=True)
        elif b_test==False:
            print('Changing name - not a test')
            preprocess.dchange_fname(d_rename,b_test=False)
        else:
            pass

def autofluorescence_subtract_dir(regdir,codedir,d_channel,ls_exclude,subdir,d_early={}):
    '''
    AF subtract images
    '''
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        print(s_dir)
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        #preprocess.cmif_mkdir([f'{s_path}/AFSubtracted'])
        s_sample = s_dir.split('-Scene')[0]
        df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
        #load exposure times csv
        df_exp = pd.read_csv(f'{codedir}/{s_sample}_ExposureTimes.csv',index_col=0,header=0)#
        #AF subtract images
        df_img_exp = mpimage.add_exposure(df_img,df_exp,type='czi')
        if len(d_early)>0:
            df_markers, df_copy = mpimage.subtract_scaled_images(df_img_exp,d_late=d_channel,
                d_early=d_early, ls_exclude=ls_exclude,subdir=subdir,b_8bit=False)
        else:
            df_markers, df_copy = mpimage.subtract_images(df_img_exp,d_channel=d_channel,
                ls_exclude=ls_exclude,subdir=subdir,b_8bit=False)

    return(df_markers)

def autofluorescence_subtract(s_sample,df_img,codedir,d_channel,ls_exclude,subdir,d_early={}):
    '''
    AF subtract images
    '''
    df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
    #load exposure times csv
    df_exp = pd.read_csv(f'{codedir}/{s_sample}_ExposureTimes.csv',index_col=0,header=0)#
    #AF subtract images
    df_img_exp = mpimage.add_exposure(df_img,df_exp,type='czi')
    if len(d_early)>0:
            df_markers, df_copy = mpimage.subtract_scaled_images(df_img_exp,d_late=d_channel,
                d_early=d_early, ls_exclude=ls_exclude,subdir=subdir,b_8bit=False)
    else:
            df_markers, df_copy = mpimage.subtract_images(df_img_exp,d_channel=d_channel,
                ls_exclude=ls_exclude,subdir=subdir,b_8bit=False)

    return(df_markers)

def multipage_ome_tiff(d_combos,d_crop,tu_dim,s_dapi,regdir,b_crop=False):
    '''
    make custom overlays, either original of AF subtracted, save at 8 bit for size, and thresholding
    '''
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        print(s_dir)
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.parse_org(s_end = "ORG.tif",s_start='R',type='reg')
        df_dapi =  df_img[df_img.marker.str.contains(s_dapi.split('_')[0])]
        df_img_stain = df_img[(~df_img.marker.str.contains('DAPI'))]
        #check
        es_test = set()
        for key, item in d_combos.items():
            es_test = es_test.union(item)
        print(set(df_img_stain.marker) - es_test)

        #cropped
        if b_crop:
            s_scene = set(d_crop).intersection(set(df_img.scene))
            d_crop_scene={k: d_crop[k] for k in (sorted(s_scene))}
            process.custom_crop_overlays(d_combos,d_crop_scene, df_img,s_dapi, tu_dim=tu_dim) #df_dapi,
        else:
            process.custom_overlays(d_combos, df_img_stain, df_dapi)

def visualize_multicolor_overlay(s_scene,subdir,qcdir,d_overlay,d_crop,es_bright,high_thresh):
    s_sample = s_scene.split('-Scene')[0]
    preprocess.cmif_mkdir([f'{qcdir}/{s_sample}'])
    if os.path.exists(f'{subdir}/{s_sample}'):
        s_path = f'{subdir}/{s_sample}'
    elif os.path.exists(f'{subdir}/{s_scene}'):
        s_path = f'{subdir}/{s_scene}'
    os.chdir(s_path)
    df_img = mpimage.parse_org()
    df_img['path'] = [f'{s_path}/{item}' for item in df_img.index]
    df_dapi_round = df_img[(df_img.color=='c1')&(df_img.scene==s_scene) & (df_img.rounds=='R2')]
    df_scene = df_img[(df_img.color!='c1') & (df_img.scene==s_scene)]
    for s_round,ls_marker  in d_overlay.items():
        print(f'Generating multicolor overlay {[item for item in ls_marker]}')
        df_round = df_scene[df_scene.marker.isin(ls_marker)]
        high_thresh=0.999
        d_overlay_round = {s_round:ls_marker}
        d_result = mpimage.multicolor_png(df_round,df_dapi_round,s_scene=s_scene,d_overlay=d_overlay_round,d_crop=d_crop,es_dim={'nada'},es_bright=es_bright,low_thresh=2000,high_thresh=high_thresh)
        for key, tu_result in d_result.items():
            io.imsave(f'{qcdir}/{s_sample}/ColorArray_{s_scene}_{key}_{".".join(tu_result[0])}.png',tu_result[1])

def cropped_ometiff(s_scene,subdir,cropdir,d_crop,d_combos,s_dapi,tu_dim,b_8bit=True):
    s_sample = s_scene.split('-Scene')[0]
    if os.path.exists(f'{subdir}/{s_sample}'):
        os.chdir(f'{subdir}/{s_sample}')
    elif os.path.exists(f'{subdir}/{s_scene}'):
        os.chdir(f'{subdir}/{s_scene}')
    df_img = mpimage.parse_org()
    d_crop_scene = {s_scene:d_crop[s_scene]}
    if b_8bit:
        dd_result = mpimage.overlay_crop(d_combos,d_crop_scene,df_img,s_dapi,tu_dim)
    else:
        dd_result = mpimage.overlay_crop(d_combos,d_crop_scene,df_img,s_dapi,tu_dim,b_8bit=False)
    for s_crop, d_result in dd_result.items():
        for s_type, (ls_marker, array) in d_result.items():
            print(f'Generating multi-page ome-tiff {[item for item in ls_marker]}')
            new_array = array[np.newaxis,np.newaxis,:]
            s_xml =  ometiff.gen_xml(new_array, ls_marker)
            with tifffile.TiffWriter(f'{cropdir}/{s_crop}_{s_type}.ome.tif') as tif:
                tif.save(new_array,  photometric = "minisblack", description=s_xml, metadata = None)

def crop_registered(s_scene,bigdir,regdir,d_crop):
    '''
    crop a stack of tiffs to the specified coordinates
    d_crop: crop to scene:(xmin, y_min, xmax, ymax)
    '''
    s_sample = s_scene.split('-Scene')[0]
    print(s_scene)
    os.chdir(f'{bigdir}/{s_scene}')
    df_img = mpimage.parse_org()
    df_scene = df_img[df_img.scene==s_scene]
    for s_image in df_scene.index:
        #print(s_image)
        a_dapi = io.imread(s_image)
        for idx, xy_cropcoor in d_crop.items():
            #crop 
            a_crop = a_dapi[xy_cropcoor[1]:xy_cropcoor[3],xy_cropcoor[0]:xy_cropcoor[2]]
            preprocess.cmif_mkdir([f'{regdir}/{s_sample}-Scene-{idx:03}'])
            io.imsave(f'{regdir}/{s_sample}-Scene-{idx:03}/{s_image.replace(s_scene,f"{s_sample}-Scene-{idx:03}")}',a_crop,check_contrast=False)

def multipage_tiff(d_combos,d_crop,tu_dim,s_dapi,regdir,b_crop=False):
    '''
    make custom overlays, either original of AF subtracted, save at 8 bit for size, and thresholding
    '''
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        print(s_dir)
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.parse_org(s_end = "ORG.tif",s_start='R',type='reg')
        df_dapi =  df_img[df_img.marker.str.contains(s_dapi.split('_')[0])]
        df_img_stain = df_img[(~df_img.marker.str.contains('DAPI'))]
        #check
        es_test = set()
        for key, item in d_combos.items():
            es_test = es_test.union(item)
        print(set(df_img_stain.marker) - es_test)

        #cropped
        if b_crop:
            s_scene = set(d_crop).intersection(set(df_img.scene))
            d_crop_scene={k: d_crop[k] for k in (sorted(s_scene))}
            process.custom_crop_overlays(d_combos,d_crop_scene, df_img,s_dapi, tu_dim=tu_dim) #df_dapi,
        else:
            process.custom_overlays(d_combos, df_img_stain, df_dapi)

def crop_basins(d_crop,tu_dim,segdir,cropdir,s_type='Cell'):
    """
    crop the segmentation basins (cell of nuceli) to same coord as images for veiwing in Napari
    """
    cwd = os.getcwd()
    for s_scene, xy_cropcoor in d_crop.items():
        print(s_scene)
        s_sample = s_scene.split('-Scene-')[0]
        os.chdir(f'{segdir}/{s_sample}_Segmentation/')

        for s_file in os.listdir():
            if s_file.find(f'{s_type} Segmentation Basins.tif') > -1: #Nuclei Segmentation Basins.tif #Cell Segmentation Basins.tif
                if s_file.find(s_scene.split('-Scene-')[1]) > -1:
                    a_seg = skimage.io.imread(s_file)
                    a_crop = a_seg[(xy_cropcoor[1]):(xy_cropcoor[1]+tu_dim[1]),(xy_cropcoor[0]):(xy_cropcoor[0]+tu_dim[0])]
                    s_coor = f'x{xy_cropcoor[0]}y{xy_cropcoor[1]}.tif'
                    #crop file
                    s_file_new = f'{cropdir}/{s_sample}-{s_file.replace(" - ","_").replace(" ","").replace("Scene","Scene-").replace(".tif",s_coor)}'
                    print(s_file_new)
                    skimage.io.imsave(s_file_new,a_crop)
    os.chdir(cwd)

def load_crop_labels(d_crop,tu_dim,segdir,cropdir,s_find='Nuclei Segmentation Basins'):
    """
    crop the segmentation basins (cell of nuceli) to same coord as images for veiwing in Napari
    s_find: 'exp5_CellSegmentationBasins' or 'Nuclei Segmentation Basins'
    """
    cwd = os.getcwd()
    for s_scene, xy_cropcoor in d_crop.items():
        print(s_scene)
        s_sample = s_scene.split('-Scene-')[0]
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation/')

        for s_file in os.listdir():
            if s_file.find(s_find) > -1: #Nuclei Segmentation Basins.tif #Cell Segmentation Basins.tif
                if s_file.find(s_scene.split(s_sample)[1]) > -1:
                    a_seg = skimage.io.imread(s_file)
                    a_crop = a_seg[(xy_cropcoor[1]):(xy_cropcoor[1]+tu_dim[1]),(xy_cropcoor[0]):(xy_cropcoor[0]+tu_dim[0])]
                    s_coor = f'x{xy_cropcoor[0]}y{xy_cropcoor[1]}.tif'
                    #crop file
                    s_file_new = f'{cropdir}/{s_file.replace(" ","").replace(".tif",s_coor)}'
                    print(s_file_new)
                    skimage.io.imsave(s_file_new,a_crop)
    os.chdir(cwd)

def load_labels(d_crop,segdir,s_find='Nuclei Segmentation Basins'):
    """
    load the segmentation basins (cell of nuceli) 
    s_find: 'exp5_CellSegmentationBasins' or 'Nuclei Segmentation Basins'
    """
    d_label={}
    cwd = os.getcwd()
    for s_scene, xy_cropcoor in d_crop.items():
        print(s_scene)
        s_sample = s_scene.split('-Scene-')[0]
        os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation/')
        for s_file in os.listdir():
            if s_file.find(s_find) > -1: #Nuclei Segmentation Basins.tif #Cell Segmentation Basins.tif
                if s_file.find(s_scene.split(s_sample)[1]) > -1:
                    a_seg = skimage.io.imread(s_file)
                    d_label.update({s_scene:a_seg})
    os.chdir(cwd)
    return(d_label)

def crop_labels(d_crop,d_label,tu_dim,cropdir,s_name='Nuclei Segmentation Basins'):
    """
    crop the segmentation basins (cell of nuceli) to same coord as images for veiwing in Napari
    s_name = 
    """
    for s_scene, xy_cropcoor in d_crop.items():
        print(s_scene)
        a_seg = d_label[s_scene]
        a_crop = a_seg[(xy_cropcoor[1]):(xy_cropcoor[1]+tu_dim[1]),(xy_cropcoor[0]):(xy_cropcoor[0]+tu_dim[0])]
        s_coor = f'x{xy_cropcoor[0]}y{xy_cropcoor[1]}.tif'
        #crop file
        s_file_new = f'{cropdir}/{s_name.replace(" ","").replace(".tif",s_coor)}'
        print(s_file_new)
        skimage.io.imsave(s_file_new,a_crop)


#### OLD: for Guillaume's pipeline ###

def copy_files(dir,dapi_copy, marker_copy,b_test=True):
    """
    copy and rename files if needed as dummies
    """
    os.chdir(dir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{dir}/{s_dir}'
        os.chdir(s_path)
        s_sample = s_dir.split('-Scene')[0]
        df_img = mpimage.parse_org(s_end = "ORG.tif")
        print(s_dir)
        if b_test:
            for key, dapi_item in dapi_copy.items():
                preprocess.copy_dapis(s_r_old=key,s_r_new=f'-R{dapi_item}_',s_c_old='_c1_',s_c_new='_c2_',s_find='_c1_ORG.tif',b_test=True)
            i_count=0
            for idx,(key, item) in enumerate(marker_copy.items()):
                preprocess.copy_markers(df_img, s_original=key, ls_copy = item,i_last_round= dapi_item + i_count, b_test=True)
                i_count=i_count + len(item)
        elif b_test==False:
            print('Changing name - not a test')
            for key, dapi_item in dapi_copy.items():
                preprocess.copy_dapis(s_r_old=key,s_r_new=f'-R{dapi_item}_',s_c_old='_c1_',s_c_new='_c2_',s_find='_c1_ORG.tif',b_test=False)
            i_count=0
            for idx,(key, item) in enumerate(marker_copy.items()):
                preprocess.copy_markers(df_img, s_original=key, ls_copy = item,i_last_round= dapi_item + i_count, b_test=False)
                i_count=i_count + len(item)
        else:
            pass

def segmentation_thresholds(regdir,qcdir, d_segment):
    """
    visualize binary mask of segmentaiton threholds
    """
    preprocess.cmif_mkdir([f'{qcdir}/Segmentation'])
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
        s_sample = s_dir.split('-Scene')[0]
        print(s_sample)
        if  (len(set(df_img.scene))) < 3:
            d_seg = preprocess.check_seg_markers(df_img,d_segment, i_rows=1, t_figsize=(10,6)) #few scenes
        elif  (len(set(df_img.scene))) > 8:
            d_seg = preprocess.check_seg_markers(df_img,d_segment, i_rows=3, t_figsize=(10,6)) #more scenes
        else:
            d_seg = preprocess.check_seg_markers(df_img,d_segment, i_rows=2, t_figsize=(10,6)) #more scenes
        for key, fig in d_seg.items():
            fig.savefig(f'{qcdir}/Segmentation/{s_dir}_{key}_segmentation.png')

def move_af_img(s_sample, regdir, subdir, dirtype='tma',b_move=False):
    '''
    dirtype = 'single' or 'tma' or 'unsub'
    '''
    #move
    os.chdir(regdir)
    for s_dir in sorted(os.listdir()):
        if s_dir.find(s_sample)>-1:
            if dirtype =='single':
                preprocess.cmif_mkdir([f'{subdir}/{s_dir}'])
            elif dirtype == 'tma':
                preprocess.cmif_mkdir([f'{subdir}/{s_sample}'])
            elif dirtype == 'unsub':
                preprocess.cmif_mkdir([f'{subdir}/{s_sample}'])
            if dirtype != 'unsub':
                print(f'{regdir}/{s_dir}/AFSubtracted')
                os.chdir(f'{regdir}/{s_dir}/AFSubtracted')
            else:
                os.chdir(f'{regdir}/{s_dir}')
            for s_file in sorted(os.listdir()):
                    if dirtype =='single':
                        movedir = f'{subdir}/{s_dir}/{s_file}'
                        print(f'{regdir}/{s_dir}/AFSubtracted/{s_file} moved to {movedir}')
                    elif dirtype == 'tma':
                        movedir = f'{subdir}/{s_sample}/{s_file}'
                        print(f'{regdir}/{s_dir}/AFSubtracted/{s_file} moved to {movedir}')
                    elif dirtype == 'unsub':
                        movedir = f'{subdir}/{s_sample}/{s_file}'
                        print(f'{regdir}/{s_dir}/{s_file} moved to {movedir}')
                    if b_move:
                        if dirtype != 'unsub':
                            shutil.move(f'{regdir}/{s_dir}/AFSubtracted/{s_file}', f'{movedir}')
                        else:
                            shutil.move(f'{regdir}/{s_dir}/{s_file}', f'{movedir}')

def extract_dataframe(s_sample, segdir,qcdir,i_rows=1):
    '''
    get mean intensity, centroid dataframes
    '''
    preprocess.cmif_mkdir([f'{qcdir}/Segmentation'])
    #get data
    os.chdir(segdir)
    dd_run = getdata.get_df(s_folder_regex=f"^{s_sample}.*_Features$",es_value_label = {"MeanIntensity","CentroidY","CentroidX"})#
    os.chdir(f'{s_sample}_Segmentation')
    d_reg = process.check_seg(s_sample=s_sample,ls_find=['Cell Segmentation Full Color'], i_rows=i_rows, t_figsize=(8,8))#
    for key, item in d_reg.items():
        item.savefig(f'{qcdir}/Segmentation/FullColor_{key}.png')

def metadata_table(regdir,segdir):
    """
    output channel/marker mapping
    """
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
        if len(set(df_img.scene)) > 1:
            df_img = df_img[df_img.scene==sorted(set(df_img.scene))[1]]
            s_sample = s_dir
        else:
            s_sample = s_dir.split('-Scene')[0]
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
        df_dapi.to_csv(f'{segdir}/metadata_{s_sample}_RoundsCyclesTable.csv',header=True)

def segmentation_inputs(regdir,segdir, d_segment,tma_bool=False,b_start=False,i_counter=0,b_java=False):
    """
    make inputs for guillaumes segmentation
    """

    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        s_path = f'{regdir}/{s_dir}'
        os.chdir(s_path)
        df_img = mpimage.parse_org(s_end = "ORG.tif",type='reg')
        if len(set(df_img.scene)) > 1:
            df_img = df_img[df_img.scene==sorted(set(df_img.scene))[1]]
            s_sample = s_dir
        else:
            s_sample = s_dir.split('-Scene')[0]
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
        df_dapi.to_csv(f'{segdir}/metadata_{s_sample}_RoundsCyclesTable.csv',header=True)
        #create cluster.java file
        if b_java:
            df_dapi.to_csv('RoundsCyclesTable.txt',sep=' ',header=False)
            preprocess.cluster_java(s_dir=f'JE{idx + i_counter}',s_sample=s_sample,imagedir=f'{s_path}',segmentdir=segdir,type='exacloud',b_segment=True,b_TMA=tma_bool)
        if b_start:
            os.chdir(f'{s_work_path}/exacloud/JE{idx}') #exacloud
            #shutil.copyfile(f'{s_src_path}/src/javawrapper.sh', './javawrapper.sh')
            print(f'JE{idx + i_counter}')
            subprocess.run(["make"])
            subprocess.run(["make", "slurm"])

def prepare_dataframe(s_sample,ls_dapi,dapi_thresh,d_channel,ls_exclude,segdir,codedir,s_af='none', b_afsub=False):
    '''
    filter data by last dapi, standard location, subtract AF, output treshold csv
    ls_dapi[0] becomes s_dapi
    '''

    os.chdir(f'{segdir}')
    #load data
    df_mi = process.load_mi(s_sample)
    df_xy = process.load_xy(s_sample)
    #drop extra centroid columns,add scene column
    df_xy = df_xy.loc[:,['DAPI_X','DAPI_Y']]
    df_xy = process.add_scene(df_xy)
    df_xy.to_csv(f'features_{s_sample}_CentroidXY.csv')
    #filter by last DAPI
    df_dapi_mi = process.filter_dapi(df_mi,df_xy,ls_dapi[0],dapi_thresh,b_images=True)

    #filter mean intensity by biomarker location in metadata
    df_filter_mi, es_standard = process.filter_standard(df_dapi_mi,d_channel,s_dapi=ls_dapi[0])

    df_filter_mi.to_csv(f'features_{s_sample}_FilteredMeanIntensity_{ls_dapi[0]}{dapi_thresh}.csv')
    #background qunatiles
    '''
    df_bg = process.filter_background(df_mi, es_standard)
    df_bg.to_csv(f'features_{s_sample}_BackgroundQuantiles.csv')
    df_bg = process.filter_background(df_dapi_mi, es_standard)
    df_bg.to_csv(f'features_{s_sample}_FilteredBackgroundQuantiles.csv')

    df_t = pd.read_csv(f'metadata_{s_sample}_RoundsCyclesTable.csv',index_col=0,header=0)
    df_exp = pd.read_csv(f'{codedir}/{s_sample}_ExposureTimes.csv',index_col=0,header=0)
    df_tt = process.add_exposure_roundscyles(df_t, df_exp,es_standard, ls_dapi = ls_dapi)
    df_tt.to_csv(f'metadata_{s_sample}_RoundsCyclesTable_ExposureTimes.csv')
    if b_afsub:
        #load metadata
        df_t = pd.read_csv(f'metadata_{s_sample}_RoundsCyclesTable_ExposureTimes.csv',index_col=0,header=0)
        #normalize by exposure time, and save to csv
        lb_columns = [len(set([item]).intersection(set(df_t.index)))>0 for item in [item.split('_')[0] for item in df_filter_mi.columns]]
        df_filter_mi = df_filter_mi.loc[:,lb_columns]
        df_norm = process.exposure_norm(df_filter_mi,df_t)
        df_norm.to_csv(f'features_{s_sample}_ExpNormalizedMeanIntensity_{ls_dapi[0]}{dapi_thresh}.csv')
        #subtract AF channels in data
        df_sub,ls_sub,ls_record = process.af_subtract(df_norm,df_t,d_channel,ls_exclude)
        df_out = process.output_subtract(df_sub,df_t)
        df_sub.to_csv(f'features_{s_sample}_AFSubtractedMeanIntensityNegative{s_af}_{ls_dapi[0]}{dapi_thresh}.csv')
        df_out.to_csv(f'features_{s_sample}_AFSubtractedMeanIntensity{s_af}_{ls_dapi[0]}{dapi_thresh}.csv')
        f = open(f"{s_sample}_AFsubtractionData_{s_af}.txt", "w")
        f.writelines(ls_record)
        f.close()
    else:
        df_out = df_filter_mi
    #output thresholding csv
    #df_out = process.add_scene(df_out) #df_out
    #df_thresh = process.make_thresh_df(df_out,ls_drop=None)
    #df_thresh.to_csv(f'thresh_XX_{s_sample}.csv')
    '''
    print('Done')

def fetch_celllabel(s_sampleset, s_slide, s_ipath, s_opath = './', es_scene = None, es_filename_endswith ={'Cell Segmentation Basins.tif', 'Nuclei Segmentation Basins.tif'}, s_sep = ' - ', b_test=True):
    '''
    input:
        s_sampleset: sample set name. e.g. jptma
        s_slide: slide name. e.g. jp-tma1-1
        es_scene: set of scenes of interest. The scenes have to be written in the same way as in the basin file name.
            if None, all scenes are if interest. default is None.
        s_ipath: absolute or relative path where the basin files can be found.
        s_opath: path to where the fetched basin files should be outputed.
            a folder, based on the s_sampleset, will be generated (if it not already exist), where the basin files will be placed.
        es_filename_endswith: set of patters that defind the endings of the files of interest.
        s_sep: separator to separate slide and scenes in the file name.
        b_test: test flag. if True no files will be copied, it is just a simulation mode.

    output:
        folder with basin flies. placed at {s_opath}{s_sampleset}_segmentation_basin/

    description:
      fetches basin (cell label) files from Guillaume's segmentation pipeline
      and copies them into a folder at s_opath, named according to s_sampleset name.
    '''
    # generate output directory
    os.makedirs('{}{}_segmentation_basin/'.format(s_opath, s_sampleset), exist_ok=True)
    # processing
    if (es_scene is None):
        i_total = 'all'
    else:
        i_total = len(es_scene) * len(es_filename_endswith)
        es_sanity_scene = copy.deepcopy(es_scene)
    i = 0
    for s_file in sorted(os.listdir(s_ipath)):
        # check for file of interest
        b_flag = False
        for s_filename_endswith in es_filename_endswith:
            if (s_file.endswith(s_filename_endswith)):
                if (es_scene is None):
                    b_flag = True
                    break
                else:
                    for s_scene in es_scene:
                        if (s_file.startswith(s_scene + s_sep)):
                            es_sanity_scene.discard(s_scene)
                            b_flag = True
                            break
                break
        # copy file
        if (b_flag):
            i += 1
            print('copy {}/{}: {}{}{} ...'.format(i, i_total, s_slide, s_sep, s_file))
            if not (b_test):
                shutil.copyfile(src='{}{}'.format(s_ipath, s_file), dst='{}{}_segmentation_basin/{}{}{}'.format(s_opath, s_sampleset, s_slide, s_sep, s_file))
    # sanity check
    if not (es_scene is None) and (i != i_total):
        sys.exit('Error: no file found for es_scene specified scene {}'.format(sorted(es_sanity_scene)))