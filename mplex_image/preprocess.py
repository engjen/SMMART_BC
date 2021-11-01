####
# title: preprocess.py
#
# language: Python3.6
# date: 2019-06-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 library to prepare images and other inputs for guillaumes segmentation software
####

#libraries
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import shutil
import re

#set src path (CHANGE ME)
s_src_path = '/home/groups/graylab_share/OMERO.rdsStore/engje/Data/cmIF'
s_work_path = '/home/groups/graylab_share/Chin_Lab/ChinData/Work/engje'

# function
# import importlib
# importlib.reload(preprocess)

def check_names(df_img,s_type='tiff'):
    """
    (CHANGE ME)
    Based on filenames in segment folder, 
    checks marker names against standard list of biomarkers
    returns a dataframe with Rounds Cycles Info, and sets of wrong and correct names
    Input: s_find = string that will be unique to one scene to check in the folder
    """
    if s_type == 'tiff':
        es_names = set(df_img.marker)
    elif s_type == 'czi':
        lls_marker =  [item.split('.') for item  in df_img.markers]
        es_names =  set([item for sublist in lls_marker for item in sublist])
    else :
        print('Unknown type')
    es_standard = {'DAPI','PDL1','pERK','CK19','pHH3','CK14','Ki67','Ecad','PCNA','HER2','ER','CD44',
        'aSMA','AR','pAKT','LamAC','CK5','EGFR','pRB','FoxP3','CK7','PDPN','CD4','PgR','Vim',
        'CD8','CD31','CD45','panCK','CD68','PD1','CD20','CK8','cPARP','ColIV','ColI','CK17',
        'H3K4','gH2AX','CD3','H3K27','53BP1','BCL2','GRNZB','LamB1','pS6RP','BAX','RAD51',
        'R0c2','R0c3','R0c4','R0c5','R5Qc2','R5Qc3','R5Qc4','R5Qc5','R11Qc2','R11Qc3','R11Qc4','R11Qc5',
        'R7Qc2','R7Qc3','R7Qc4','R7Qc5','PDL1ab','PDL1d','R14Qc2','R14Qc3','R14Qc4','R14Qc5',
        'R8Qc2','R8Qc3','R8Qc4','R8Qc5','R12Qc2','R12Qc3','R12Qc4','R12Qc5','PgRc4','R1c2','CCND1',
        'Glut1','CoxIV','LamB2','S100','BMP4','BMP2','BMP6','pS62MYC', 'CGA', 'p63', 'SYP','PDGFRa', 'HIF1a','CC3',
        'MUC1','CAV1','MSH2','CSF1R','R13Qc4', 'R13Qc5', 'R13Qc3', 'R13Qc2','R10Qc2','R10Qc3','R10Qc4','R10Qc5',
        'R6Qc2', 'R6Qc3','R6Qc4', 'R6Qc5', 'TUBB3', 'CD90', 'GATA3'}#,'PDGFRB'CD66b (Neutrophils)
        #HLA class II or CD21(Dendritic cells)
        #BMP4	Fibronectin, CD11b (dendritic, macrophage/monocyte/granulocyte)	CD163 (macrophages)
        #CD83 (dendritic cells)	FAP	
    es_wrong = es_names - es_standard
    es_right = es_standard.intersection(es_names)
    print(f'Wrong names {es_wrong}')
    print(f' Right names {es_right}')
    return(es_wrong)

def copy_dapis(s_r_old='-R11_',s_r_new='-R91_',s_c_old='_c1_',s_c_new='_c2_',s_find='_c1_ORG.tif',b_test=True,type='org'):
    """
    copy specified round of dapi, rename with new round and color
    Input:
    s_r_old = old round
    s_r_new = new round on copied DAPI
    s_c_old = old color
    s_c_new = new color on copied DAPI
    s_find= how to identify dapis i.e. '_c1_ORG.tif'
    b_test=True if testing only
    """
    i_dapi = re.sub("[^0-9]", "", s_r_old)
    ls_test = []
    for s_file in os.listdir():
            if s_file.find(s_find) > -1:
                if s_file.find(s_r_old) > -1:
                    s_file_round = s_file.replace(s_r_old,s_r_new)
                    s_file_color = s_file_round.replace(s_c_old,s_c_new)
                    if type=='org':
                        s_file_dapi = s_file_color.replace(s_file_color.split("_")[1],f'DAPI{i_dapi}.DAPI{i_dapi}.DAPI{i_dapi}.DAPI{i_dapi}')
                    else:
                        s_file_dapi=s_file_color
                    ls_test = ls_test + [s_file]
                    if b_test:
                        print(f'copied file {s_file} \t and named {s_file_dapi}')
                    else:
                        print(f'copied file {s_file} \t and named {s_file_dapi}')
                        shutil.copyfile(s_file, s_file_dapi)
    
    print(f'total number of files changed is {len(ls_test)}')

def copy_markers(df_img, s_original = 'panCK', ls_copy = ['CK19','CK5','CK7','CK14'],i_last_round = 97, b_test=True, type = 'org'):
    """
    copy specified marker image, rename with new round and color (default c2) and marker name
    Input:
    s_original = marker to copy
    df_img = dataframe with images
    ls_copy = list of fake channels to make

    b_test=True if testing only
    """
    df_copy = df_img[df_img.marker==s_original]
    ls_test = []
    for s_index in df_copy.index:
            s_round = df_img.loc[s_index,'rounds']
            for idx, s_copy in enumerate(ls_copy):
                i_round = i_last_round + 1 + idx
                s_round = df_img.loc[s_index,'rounds']
                s_roundnum = re.sub("[^0-9]", "", s_round)
                s_round_pre = s_round.replace(s_roundnum,'')
                s_file_round = s_index.replace(df_img.loc[s_index,'rounds'],f'{s_round_pre}{i_round}')
                s_file_color = s_file_round.replace(f'_{s_round}_',f'_c{i_round}_')
                if type == 'org':
                    s_file_dapi = s_file_color.replace(s_file_color.split("_")[1],f'{s_copy}.{s_copy}.{s_copy}.{s_copy}')
                else:
                    s_file_dapi = s_file_color.replace(f'_{s_original}_',f'_{s_copy}_')
                ls_test = ls_test + [s_index]
                if b_test:
                    print(f'copied file {s_index} \t and named {s_file_dapi}')
                else:
                    print(f'copied file {s_index} \t and named {s_file_dapi}')
                    shutil.copyfile(s_index, s_file_dapi)
    print(f'total number of files changed is {len(ls_test)}')

def dchange_fname(d_rename={'_oldstring_':'_newstring_'},b_test=True):
    """
    replace anything in file name, based on dictionary of key = old
    values = new
    Input
    """
    #d_rename = {'Registered-R11_CD34.AR.':'Registered-R11_CD34.ARcst.','FoxP3b':'FoxP3bio'}
    for s_key,s_value in d_rename.items():
        s_old=s_key
        s_new=s_value
        #test
        if b_test:
            ls_test = []
            for s_file in os.listdir():
                if s_file.find(s_old) > -1:
                    s_file_print = s_file
                    ls_test = ls_test + [s_file]
                    len(ls_test)
                    s_file_new = s_file.replace(s_old,s_new)
                    #print(f'changed file {s_file}\tto {s_file_new}')
            if len(ls_test)!=0:
                print(f'changed file {s_file_print}\tto {s_file_new}')
            print(f'total number of files changed is {len(ls_test)}')
        #really rename
        else:
            ls_test = []
            for s_file in os.listdir():
                if s_file.find(s_old) > -1:
                    s_file_print = s_file
                    ls_test = ls_test + [s_file]
                    len(ls_test)
                    s_file_new = s_file.replace(s_old,s_new)
                    #print(f'changed file {s_file}\tto {s_file_new}')
                    os.rename(s_file, s_file_new) #comment out this line to test
            if len(ls_test)!=0:
                print(f'changed file {s_file_print}\tto {s_file_new}')
            print(f'total number of files changed is {len(ls_test)}')

def csv_change_fname(i_scene_len=2, b_test=True):
    '''
    give a csv with wrong_round and correct scene names
    make a Renamed folder
    the correct scene is added after, as +correct
    '''
    df_test = pd.read_csv(f'FinalSceneNumbers.csv',header=0)
    df_test = df_test.astype(str)#(works!)
    if i_scene_len == 2:
        df_scene = df_test.applymap('{:0>2}'.format)
    elif i_scene_len == 3:
        df_test.replace('nan','',inplace=True)
        df_test.replace(to_replace = "\.0+$",value = "", regex = True,inplace=True)
        df_scene = df_test.applymap('{:0>3}'.format)
    else:
        df_scene = df_test #.applymap('{:0>3}'.format)
    #for each round with wrong names
    for s_wrong in  df_scene.columns[df_scene.columns.str.contains('wrong')]:
        for s_file in os.listdir():
            #find files in that round
            if s_file.find(f'R{s_wrong.split("_")[1]}_') > -1:
                #print(s_file)
                #for each scene
                for s_index in df_scene.index:
                    s_wrong_scene = df_scene.loc[s_index,s_wrong]
                    if s_file.find(f'-Scene-{s_wrong_scene}') > -1:
                        s_correct = df_scene.loc[s_index,'correct']
                        print(s_correct)
                        s_replace = s_file.replace(f'-Scene-{s_wrong_scene}', f'-Scene-{s_wrong_scene}+{s_correct}')
                        s_file_new = f"./Renamed/{s_replace}"
                        
                        if b_test:
                            print(f'changed file {s_file} to {s_file_new}')
                        else:
                            os.rename(s_file, s_file_new)
                            print(f'changed file {s_file} to {s_file_new}')
    return(df_test)

def check_seg_markers(df_img,d_segment = {'CK19':1002,'CK5':5002,'CD45':2002,'Ecad':802,'CD44':1202,'CK7':2002,'CK14':502}, i_rows=1, t_figsize=(20,10)):
    """
    This script makes binarizedoverviews of all the specified segmentation markers
    with specified thresholds, and outputs a rounds cycles table
    Input: df_dapi: output of mpimage.parse_org()
     d_segment: segmentation marker names and thresholds
     i_rows = number or rows in figure
     t_figsize = (x, y) in inches size of figure
    Output: dictionary
    """
    d_result = {}
    for s_key,i_item in d_segment.items():
        #find all segmentation marker slides
        df_img_seg = df_img[df_img.marker==s_key]
        fig,ax = plt.subplots(i_rows,(len(df_img_seg)+(i_rows-1))//i_rows, figsize = t_figsize, squeeze=False)
        ax = ax.ravel()
        for idx,s_scene in enumerate(sorted(df_img_seg.index.tolist())):
            print(f'Processing {s_scene}')
            im_low = skimage.io.imread(s_scene)
            im = skimage.exposure.rescale_intensity(im_low,in_range=(i_item,i_item+1))
            ax[idx].imshow(im, cmap='gray')
            s_round = s_scene.split('Scene')[1].split('_')[0]
            ax[idx].set_title(f'{s_key} Scene{s_round} min={i_item}',{'fontsize':12})
        plt.tight_layout()
        d_result.update({s_key:fig})
    return(d_result)

def checkall_seg_markers(df_img,d_segment = {'CK19':1002,'CK5':5002,'CD45':2002,'Ecad':802,'CD44':1202,'CK7':2002,'CK14':502}, i_rows=2, t_figsize=(15,10)):
    """
    This script makes binarizedoverviews of all the specified segmentation markers
    with specified thresholds, and it puts all segmentation markers in one figure
    Input: df_dapi: output of mpimage.parse_org()
     d_segment: segmentation marker names and thresholds
     i_rows = number or rows in figure
     t_figsize = (x, y) in inches size of figure
    Output: dictionary
    """
    es_seg = set([s_key for s_key,i_item in d_segment.items()])
    df_img_seg = df_img[df_img.marker.isin(es_seg)]
    fig,ax = plt.subplots(i_rows,(len(es_seg)+(i_rows-1))//i_rows, figsize = t_figsize, squeeze=False)
    ax = ax.ravel()
    for idx,s_scene in enumerate(sorted(df_img_seg.index.tolist())):
            s_key = df_img.loc[s_scene].marker
            i_item = d_segment[s_key]
            print(f'Processing {s_scene}')
            im_low = skimage.io.imread(s_scene)
            im = skimage.exposure.rescale_intensity(im_low,in_range=(i_item,i_item+1))
            ax[idx].imshow(im, cmap='gray')
            s_round = s_scene.split('Scene')[1].split('_')[0]
            ax[idx].set_title(f'{s_key} Scene{s_round} min={i_item}',{'fontsize':12})
    plt.tight_layout()
        #d_result.update({s_key:fig})
    return(fig)

def rounds_cycles(s_find='-Scene-001_c', d_segment = {'CK19':1002,'CK5':5002,'CD45':4502,'Ecad':802,'CD44':1202,'CK7':2002,'CK14':502}):
    """
    Based on filenames in segment folder, makes a dataframe with Rounds Cycles Info
    """
    ls_marker = []
    df_dapi = pd.DataFrame() #(columns=['rounds','colors','minimum','maximum','exposure','refexp','location'])
    for s_name in sorted(os.listdir()):
        if s_name.find(s_find) > -1:
            s_color = s_name.split('_')[3]
            if s_color != 'c1':
                #print(s_name)
                if s_color == 'c2':
                    s_marker = s_name.split('_')[1].split('.')[0]
                elif s_color == 'c3':
                    s_marker = s_name.split('_')[1].split('.')[1]
                elif s_color == 'c4':
                    s_marker = s_name.split('_')[1].split('.')[2]
                elif s_color == 'c5':
                    s_marker = s_name.split('_')[1].split('.')[3]
                else: 
                    print('Error: unrecognized channel name')
                    s_marker = 'error'
                ls_marker.append(s_marker)
                df_marker = pd.DataFrame(index = [s_marker],columns=['rounds','colors','minimum','maximum','exposure','refexp','location'])
                df_marker.loc[s_marker,'rounds'] = s_name.split('_')[0].split('Registered-')[1]
                df_marker.loc[s_marker,'colors'] = s_name.split('_')[3]
                df_marker.loc[s_marker,'minimum'] = 1003
                df_marker.loc[s_marker,'maximum'] = 65535
                df_marker.loc[s_marker,'exposure'] = 100
                df_marker.loc[s_marker,'refexp'] = 100
                df_marker.loc[s_marker,'location'] = 'All'
                df_dapi = df_dapi.append(df_marker)
    for s_key,i_item in d_segment.items():
        df_dapi.loc[s_key,'minimum'] = i_item
    #if len(ls_marker) != len(set(df_marker.index)):
    #    print('Check for repeated biomarkers!')
    for s_marker in ls_marker:
        if (np.array([s_marker == item for item in ls_marker]).sum()) != 1:
            print('Repeated marker!/n')
            print(s_marker)

    return(df_dapi, ls_marker)

def cluster_java(s_dir='JE1',s_sample='SampleID',imagedir='PathtoImages',segmentdir='PathtoSegmentation',type='exacloud',b_segment=True,b_TMA=True):
    """
    makes specific changes to files in Jenny's Work directories to result in Cluster.java file
    s_dir = directory to make cluster.java file in
    s_sample = unique sample ID
    imagedir = full /path/to/images
    type = 'exacloud' or 'eppec' (different make file settings)
    b_TMA = True if tissue is a TMA
    b_segment = True if segmentation if being done (or False if feature extraction only)
    """
    if type=='exacloud':
        os.chdir(f'{s_work_path}/exacloud/')
        with open('TemplateExacloudCluster.java') as f:
            s_file = f.read()
    elif type=='eppec':
        os.chdir(f'{s_work_path}/eppec/')
        with open('TemplateEppecCluster.java') as f:
            s_file = f.read()
    else:
        print('Error: type must be exacloud or eppec')
    s_file = s_file.replace('PathtoImages',imagedir)
    s_file = s_file.replace('PathtoSegmentation',f'{segmentdir}/{s_sample.split("-Scene")[0]}_Segmentation/')
    s_file = s_file.replace('PathtoFeatures',f'{segmentdir}/{s_sample.split("-Scene")[0]}_Features/')
    if b_segment:
        s_file = s_file.replace('/*cif.Experiment','cif.Experiment')
        s_file = s_file.replace('("Segmentation Done!") ;*/','("Segmentation Done!") ;')
    if b_TMA:
        s_file = s_file.replace('cif.CROPS ;','cif.TMA ;')
    os.chdir(f'./{s_dir}/')
    with open('Cluster.java', 'w') as f:
        f.write(s_file)

def registration_matlab(N_smpl='10000',N_colors='5',s_rootdir='PathtoImages',s_subdirname='RegisteredImages/',s_ref_id='./R1_*_c1_ORG.tif',
    ls_order = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R0','R11Q']):

    """
    makes specific changes to template matlab scripts files in Jenny's directories to result in .m file
    Input:
    N_smpl = i_N_smpl; %number of features to detect in image (default = 10000)
    N_colors = i_N_colors; %number of colors in R1 (default = 5)
    ls_order = {RoundOrderString}; %list of names and order of rounds
    s_rootdir = 'PathtoImages' %location of raw images in folder
    s_ref_id = 'RefDapiUniqueID'; %shared unique identifier of reference dapi
    s_subdirname = 'PathtoRegisteredImages' %location of folder where registered images will reside
    """
    ls_order_q = [f"'{item}'" for item in ls_order]
    #find template, open ,edit
    os.chdir(f'{s_src_path}/src')
    with open('template_registration_server_multislide_roundorder_scenes_2019_11_11.m') as f:
            s_file = f.read()
    s_file = s_file.replace('PathtoImages',s_rootdir)
    s_file = s_file.replace('PathtoRegisteredImages',s_subdirname)
    s_file = s_file.replace('i_N_smpl',N_smpl)
    s_file = s_file.replace('i_N_colors',N_colors)
    s_file = s_file.replace("RoundOrderString",",".join(ls_order_q))
    s_file = s_file.replace('RefDapiUniqueID',s_ref_id)

    #save edited .m file
    os.chdir(s_rootdir)
    with open('registration_py.m', 'w') as f:
        f.write(s_file)

def large_registration_matlab(N_smpl='10000',N_colors='5',s_rootdir='PathtoImages',s_subdirname='RegisteredImages',s_ref_id='./R1_*_c1_ORG.tif',
     ls_order = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R0','R11Q'],d_crop_regions={1:'[0 0 1000 1000]'}):
    """
    makes specific changes to template matlab scripts files in Jenny's directories to result in .m file
    Input:
    N_smpl = i_N_smpl; %number of features to detect in image (default = 10000)
    N_colors = i_N_colors; %number of colors in R1 (default = 5)
    ls_order = {RoundOrderString}; %list of names and order of rounds
    s_rootdir = 'PathtoImages' %location of raw images in folder
    s_ref_id = 'RefDapiUniqueID'; %shared unique identifier of reference dapi
    s_subdirname = 'PathtoRegisteredImages' %location of folder where registered images will reside
    d_crop_regions= dictioanr with crop integer as key, ans string with crop array as value e.g. {1:'[0 0 1000 1000]'}

    """
    ls_order_q = [f"'{item}'" for item in ls_order]

    os.chdir(f'{s_src_path}/src')
    with open('template_registration_server_largeimages_roundorder_2019_11_11.m') as f:
        s_file = f.read()
    s_file = s_file.replace('PathtoImages',s_rootdir)
    s_file = s_file.replace('PathtoRegisteredImages',s_subdirname)
    s_file = s_file.replace('i_N_smpl',N_smpl)
    s_file = s_file.replace('i_N_colors',N_colors)
    s_file = s_file.replace("RoundOrderString",",".join(ls_order_q))
    s_file = s_file.replace('RefDapiUniqueID',s_ref_id)

    for i_crop_region, s_crop in d_crop_regions.items():
        s_file = s_file.replace(f'%{i_crop_region}%{i_crop_region}%','')
        s_file = s_file.replace(f'[a_crop_{i_crop_region}]',s_crop)
    #save edited .m file
    os.chdir(s_rootdir)
    with open('registration_py.m', 'w') as f:
        f.write(s_file)

def cmif_mkdir(ls_dir):
    '''
    check if directories existe. if not, make them
    '''
    for s_dir in ls_dir:
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)

######################### Old functions ############################

def check_reg_channels(ls_find=['c1_ORG','c2_ORG'], i_rows=2, t_figsize=(20,10), b_separate = False, b_mkdir=True):
    """
    This script makes overviews of all the specified channel images of registered tiff images
    in a big folder (slides prepared for segmentation for example)
    Input: ls_find = list of channels to view
     i_rows = number or rows in figure
     t_figsize = (x, y) in inches size of figure
     b_mkdir = boolean whether to make a new Check_Registration folder
    Output: dictionary with {slide_color:number of rounds found}
     images of all rounds of a certain slide_color
    """
    d_result = {}
    ls_error = []
    if b_separate:
        s_dir = os.getcwd()
        os.chdir('..')
        s_path = os.getcwd()
        if b_mkdir:
            os.mkdir(f'./Check_Registration')
        os.chdir(s_dir)
    else:
        s_path = os.getcwd()
        if b_mkdir:
            os.mkdir(f'./Check_Registration')
    for s_find in ls_find:
        #find all dapi slides
        ls_dapis = []
        for s_dir in os.listdir():
            if s_dir.find(s_find) > -1:
                ls_dapis = ls_dapis + [s_dir]
        
        #find all unique scenes
        ls_scene_long = []
        for s_dapi in ls_dapis:
            ls_scene_long = ls_scene_long + [(s_dapi.split('_')[2])]
        ls_scene = list(set(ls_scene_long))
        ls_scene.sort()

        for s_scene in ls_scene:
            print(f'Processing {s_scene}')
            ls_dapi = []
            for s_file in ls_dapis:
                if s_file.find(s_scene)>-1:
                    ls_dapi = ls_dapi + [s_file]
            fig,ax = plt.subplots(i_rows,(len(ls_dapi)+(i_rows-1))//i_rows, figsize = t_figsize)
            ax = ax.ravel()
            ls_dapi.sort()
            for x in range(len(ls_dapi)):
                im_low = skimage.io.imread(ls_dapi[x])
                im = skimage.exposure.rescale_intensity(im_low,in_range=(np.quantile(im_low,0.02),np.quantile(im_low,0.98)+np.quantile(im_low,0.98)/2))
                ax[x].imshow(im, cmap='gray')
                s_round = ls_dapi[x].split('_')[0].split('-')[1]
                ax[x].set_title(s_round,{'fontsize':12})
            s_slide = ls_dapi[0].split('_')[2]
            plt.tight_layout()
            fig.savefig(f'{s_path}/Check_Registration/{s_slide}_{s_find}.png')
            d_result.update({f'{s_slide}_{s_find}':len(ls_dapi)})
            ls_error = ls_error + [len(ls_dapi)]
    if(len(set(ls_error))==1):
        print("All checked scenes/channels have the same number of images")
    else:
        print("Warning: different number of images in some scenes/channels")
        for s_key, i_item in d_result.items():
            print(f'{s_key} has {i_item} images')
    return(d_result)
	

def check_names_deprecated(s_find='-Scene-001_c',b_print=False):
    """
    Based on filenames in segment folder, 
    checks marker names against standard list of biomarkers
    returns a dataframe with Rounds Cycles Info, and sets of wrong and correct names
    Input: s_find = string that will be unique to one scene to check in the folder
    """
    df_dapi = pd.DataFrame() #(columns=['rounds','colors','minimum','maximum','exposure','refexp','location'])
    for s_name in sorted(os.listdir()):
        if s_name.find(s_find) > -1:
            s_color = s_name.split('_')[3]
            if s_color != 'c1':
                if b_print:
                    print(s_name)
                if s_color == 'c2':
                    s_marker = s_name.split('_')[1].split('.')[0]
                elif s_color == 'c3':
                    s_marker = s_name.split('_')[1].split('.')[1]
                elif s_color == 'c4':
                    s_marker = s_name.split('_')[1].split('.')[2]
                elif s_color == 'c5':
                    s_marker = s_name.split('_')[1].split('.')[3]
                else: 
                    print('Error: unrecognized channel name')
                    s_marker = 'error'
                df_marker = pd.DataFrame(index = [s_marker],columns=['rounds','colors','minimum','maximum','exposure','refexp','location'])
                df_marker.loc[s_marker,'rounds'] = s_name.split('_')[0].split('Registered-')[1]
                df_marker.loc[s_marker,'colors'] = s_name.split('_')[3]
                df_marker.loc[s_marker,'minimum'] = 1003
                df_marker.loc[s_marker,'maximum'] = 65535
                df_marker.loc[s_marker,'exposure'] = 100
                df_marker.loc[s_marker,'refexp'] = 100
                df_marker.loc[s_marker,'location'] = 'All'
                df_dapi = df_dapi.append(df_marker)
    es_names = set(df_dapi.index)
    es_standard = {'PDL1','pERK','CK19','pHH3','CK14','Ki67','Ecad','PCNA','HER2','ER','CD44',
        'aSMA','AR','pAKT','LamAC','CK5','EGFR','pRB','FoxP3','CK7','PDPN','CD4','PgR','Vim',
        'CD8','CD31','CD45','panCK','CD68','PD1','CD20','CK8','cPARP','ColIV','ColI','CK17',
        'H3K4','gH2AX','CD3','H3K27','53BP1','BCL2','GRNZB','LamB1','pS6RP','BAX','RAD51',
        'R0c2','R0c3','R0c4','R0c5','R5Qc2','R5Qc3','R5Qc4','R5Qc5','R11Qc2','R11Qc3','R11Qc4','R11Qc5',
        'R7Qc2','R7Qc3','R7Qc4','R7Qc5','PDL1ab','PDL1d','R14Qc2','R14Qc3','R14Qc4','R14Qc5',
        'R8Qc2','R8Qc3','R8Qc4','R8Qc5','R12Qc2','R12Qc3','R12Qc4','R12Qc5','PgRc4',
        'Glut1','CoxIV','LamB2','S100','BMP4','BMP2','BMP6','pS62MYC', 'CGA', 'p63', 'SYP','PDGFRa', 'HIF1a'}#,'PDGFRB'CD66b (Neutrophils)	HLA class II or CD21(Dendritic cells)
        #BMP4	Fibronectin, CD11b (dendritic, macrophage/monocyte/granulocyte)	CD163 (macrophages)
        #CD83 (dendritic cells)	FAP	Muc1
    es_wrong = es_names - es_standard
    es_right = es_standard.intersection(es_names)
    print(f'Wrong names {es_wrong}')
    print(f' Right names {es_right}')
    return(df_dapi, es_wrong, es_right)

def file_sort(s_sample, s_path, i_scenes=14,i_rounds=12,i_digits=3,ls_quench=['R5Q','R11Q'],s_find='_ORG.tif',b_scene=False):
    '''
    count rounds and channels of images (koeis naming convention, not registered yet)
    '''
    os.chdir(s_path)
    se_dir = pd.Series(os.listdir())

    se_dir = se_dir[se_dir.str.find(s_find)>-1]
    se_dir = se_dir.sort_values()
    se_dir = se_dir.reset_index()
    se_dir = se_dir.drop('index',axis=1)

    print(s_sample)
    print(f'Total _ORG.tif: {len(se_dir)}')

    #count files in each round, plus store file names on df_round
    df_round = pd.DataFrame(index=range(540))
    i_grand_tot = 0
    for x in range(i_rounds):
        se_round = se_dir[se_dir.iloc[:,0].str.contains(f'R{str(x)}_')]
        se_round = se_round.rename({0:'round'},axis=1)
        se_round = se_round.sort_values(by='round')
        se_round = se_round.reset_index()
        se_round = se_round.drop('index',axis=1)
        i_tot = se_dir.iloc[:,0].str.contains(f'R{str(x)}_').sum()
        i_round = 'Round ' + str(x)
        print(f'{i_round}: {i_tot}')
        i_grand_tot = i_grand_tot + i_tot
        df_round[i_round]=se_round
    df_round = df_round.dropna()    

    #quenched round special loop
    for s_quench in ls_quench:
        #x = "{0:0>2}".format(x)
        i_tot = se_dir.iloc[:,0].str.contains(s_quench).sum()
        #i_round = 'Round ' + str(x)
        print(f'{s_quench}: {i_tot}')
        i_grand_tot = i_grand_tot + i_tot 
    print(f'Total files containing Rxx_: {i_grand_tot}')
    
    if b_scene:
        #print number of files in each scene
        for x in range(1,i_scenes+1):
            if i_digits==3:
                i_scene = "{0:0>3}".format(x)
            elif i_digits==2:
                i_scene = "{0:0>2}".format(x)
            elif i_digits==1:
                i_scene = "{0:0>1}".format(x)
            else:
                print('wrong i_digits input (must be between 1 and 3')
            i_tot = se_dir.iloc[:,0].str.contains(f'Scene-{i_scene}_').sum()
            i_round = 'Scene ' + str(x)
            print(f'{i_round}: {i_tot}')

    #print number of files in each color
    for x in range(1,6):
        #i_scene = "{0:0>2}".format(x)
        i_tot = se_dir.iloc[:,0].str.contains(f'_c{str(x)}_ORG').sum()
        i_round = 'color ' + str(x)
        print(f'{i_round}: {i_tot}')

    d_result = {}	
    for s_round in df_round.columns:
        es_round = set([item.split('-Scene-')[1].split('_')[0] for item in list(df_round.loc[:,s_round].values)])
        d_result.update({s_round:es_round})
    print('\n')


def change_fname(s_old='_oldstring_',s_new='_newstring_',b_test=True):
    """
    replace anything in file name
    """
    if b_test:
        ls_test = []
        for s_file in os.listdir():
            if s_file.find(s_old) > -1:
                ls_test = ls_test + [s_file]
                len(ls_test)
                s_file_new = s_file.replace(s_old,s_new)
                print(f'changed file {s_file}\tto {s_file_new}')

        print(f'total number of files changed is {len(ls_test)}')
    #really rename
    else:
        ls_test = []
        for s_file in os.listdir():
            if s_file.find(s_old) > -1:
                ls_test = ls_test + [s_file]
                len(ls_test)
                s_file_new = s_file.replace(s_old,s_new)
                print(f'changed file {s_file}\tto {s_file_new}')
                os.rename(s_file, s_file_new) #comment out this line to test
        print(f'total number of files changed is {len(ls_test)}')

def check_reg_slides(i_rows=2, t_figsize=(20,10), b_mkdir=True):
    """
    This script makes overviews of all the dapi images of registered images in a big folder (slides prepared for segmentation for example)
    """
    #find all dapi slides
    ls_dapis = []
    for s_dir in os.listdir():
        if s_dir.find('c1_ORG') > -1:
            ls_dapis = ls_dapis + [s_dir]

    #find all scenes
    ls_scene_long = []
    for s_dapi in ls_dapis:
        ls_scene_long = ls_scene_long + [(s_dapi.split('Scene')[1].split('_')[0])]
    ls_scene = list(set(ls_scene_long))
    ls_scene.sort()
    if b_mkdir:
        os.mkdir(f'./Check_Registration')
    for s_scene in ls_scene:
        print(f'Processing {s_scene}')
        ls_dapi = []
        for s_file in ls_dapis:
            if s_file.find(f'Scene{s_scene}')>-1:
                ls_dapi = ls_dapi + [s_file]
        fig,ax = plt.subplots(i_rows,(len(ls_dapi)+(i_rows-1))//i_rows, figsize = t_figsize)
        ax = ax.ravel()
        ls_dapi.sort()
        for x in range(len(ls_dapi)):
            im_low = skimage.io.imread(ls_dapi[x])
            im = skimage.exposure.rescale_intensity(im_low,in_range=(np.quantile(im_low,0.02),np.quantile(im_low,0.98)+np.quantile(im_low,0.98)/2))
            ax[x].imshow(im, cmap='gray')
            s_round = ls_dapi[x].split('_')[0].split('-')[1]
            ax[x].set_title(s_round,{'fontsize':12})
        s_slide = ls_dapi[0].split('_')[2]
        plt.tight_layout()
        fig.savefig(f'Check_Registration/{s_slide}.png')

def check_reg_dirs(s_dir='SlideName',s_subdir='Registered-SlideName', i_rows=2, t_figsize=(20,10), b_mkdir=True):
    """
    this checks registration when files are in subdirectories (such as with large tissues, i.e. NP005)
    """

    rootdir = os.getcwd()
    if b_mkdir:
        os.mkdir(f'./Check_Registration')
    #locate subdirectores
    for s_dir in os.listdir():
        if s_dir.find(s_dir) > -1:
            os.chdir(f'./{s_dir}')

            #locate registered image folders
            for s_dir in os.listdir():
            #for s_dir in ls_test2:
                if s_dir.find(s_subdir) > -1:  #'Registered-BR1506-A019-Scene'
                    print(f'Processing {s_dir}')
                    ls_dapi = []
                    os.chdir(f'./{s_dir}')
                    ls_file = os.listdir()
                    for s_file in ls_file:
                        if s_file.find('_c1_ORG.tif')>-1:
                            ls_dapi = ls_dapi + [s_file]
                    fig,ax = plt.subplots(i_rows,(len(ls_dapi)+(i_rows-1))//i_rows, figsize = (t_figsize)) #vertical
                    ax=ax.ravel()
                    ls_dapi.sort()
                    for x in range(len(ls_dapi)):
                        im_low = skimage.io.imread(ls_dapi[x])
                        im = skimage.exposure.rescale_intensity(im_low,in_range=(np.quantile(im_low,0.02),np.quantile(im_low,0.98)+np.quantile(im_low,0.98)/2))
                        ax[x].imshow(im, cmap='gray')
                        s_round = ls_dapi[x].split('_')[0].split('-')[1]
                        s_scene = ls_dapi[x].split('-Scene')[1].split('_')[0]
                        ax[x].set_title(f'{s_round} Scene{s_scene}',{'fontsize':12})
                    plt.tight_layout()

                    #save figure in the rootdir/Check_Registration folder
                    fig.savefig(f'{rootdir}/Check_Registration/{s_dir}.png')
            #go out of the subfoler and start next processing
                os.chdir('..')

def test(name="this_is_you_name"):
    '''
    This is my first doc string
    '''
    print(f'hello {name}')
    return True
