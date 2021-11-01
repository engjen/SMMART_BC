# wrapper functions for codex image processing

#from mplex_image import preprocess, mpimage, process, 
from mplex_image import features 
import os
import pandas as pd
import math
import skimage
from skimage import io, filters
import re
import numpy as np

def parse_img(s_end = ".tif",s_start='reg'):
    """
    This function will parse images following akoya stiched naming convention
    """
    s_path = os.getcwd()
    ls_file = []
    for file in os.listdir():
        if file.endswith(s_end):
            if file.find(s_start)==0:
                ls_file = ls_file + [file]
    df_img = pd.DataFrame(index=ls_file)
    df_img['rounds'] = [item.split('_')[1].split('cyc')[1] for item in df_img.index]
    df_img['color'] = [item.split('_')[3] for item in df_img.index]
    df_img['slide'] = [item.split('_')[0] for item in df_img.index]
    df_img['marker'] = [item.split('_')[-1].split('.')[0] for item in df_img.index]
    df_img['marker_string'] = [item.split('_')[-1].split('.')[0] for item in df_img.index]
    df_img['path'] = [f"{s_path}/{item}" for item in df_img.index]
    return(df_img) 

def load_li(ls_sample):
    '''
    load threshold on the segmentation marker images acquired during feature extraction
    '''
    df_img_all =pd.DataFrame()
    for s_sample in ls_sample:
        df_img = pd.read_csv(f'thresh_{s_sample}_ThresholdLi.csv', index_col=0)
        df_img['rounds'] = [item.split('_')[1].split('cyc')[1] for item in df_img.index]
        df_img['color'] = [item.split('_')[3] for item in df_img.index]
        df_img['slide'] = s_sample
        df_img['scene'] = [item.split('_')[0].split('reg')[1] for item in df_img.index]
        df_img['marker'] = [item.split('_')[-1].split('.')[0] for item in df_img.index]  #parse file name for biomarker
        df_img['slide_scene'] = df_img.slide + '_scene' + df_img.scene
        df_img_all = df_img_all.append(df_img)
    return(df_img_all)

def underscore_to_dash(df_mi_full,df_img_all):
    '''
    the underscore in sample names will break downstream code; change to dash
    '''
    #naming underscore to dash
    df_mi_full['slide'] = [item.split('_scene')[0].replace('_','-') for item in df_mi_full.index]
    df_mi_full.index = [f"_scene{item.split('_scene')[1]}" for item in df_mi_full.index]
    df_mi_full.index = df_mi_full.slide + df_mi_full.index
    df_mi_full['scene'] = [item.split('_')[1] for item in df_mi_full.index]
    df_mi_full['slide_scene'] = df_mi_full.slide + '_' + df_mi_full.scene
    #df_img renameing
    df_img_all['slide'] = [item.replace('_','-') for item in df_img_all.slide]
    df_img_all['slide_scene'] = df_img_all.slide + '_scene' + df_img_all.scene
    return(df_mi_full,df_img_all)

def extract_cellpose_features(s_sample, segdir, subdir, ls_seg_markers, nuc_diam, cell_diam,s_scene='reg001'):
    '''
    load the segmentation results, the input images, and the channels images
    extract mean intensity from each image, and centroid, area and eccentricity for 
    '''

    df_sample = pd.DataFrame()
    df_thresh = pd.DataFrame()
    if os.path.exists(f'{segdir}/{s_scene}Cellpose_Segmentation'):
        os.chdir(f'{segdir}/{s_scene}Cellpose_Segmentation')
    else:
        os.chdir(f'{segdir}')
    ls_scene = []
    d_match = {}
    for s_file in os.listdir():
        if s_file.find(f'{".".join(ls_seg_markers)} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split('_')[0])
            d_match.update({s_file.split('_')[0]:s_file})
        elif s_file.find(f'{".".join(ls_seg_markers)} nuc{nuc_diam} matchedcell{cell_diam} - Cell Segmentation Basins')>-1:
            ls_scene.append(s_file.split('_')[0])
            d_match.update({s_file.split('_')[0]:s_file})
    for s_scene in ['reg001']:  #ls_scene: #one scene
        print(f'processing {s_scene}')
        for s_file in os.listdir():
            if s_file.find(s_scene) > -1:
                if s_file.find("DAPI.png") > -1:
                    s_dapi = s_file
        dapi = io.imread(s_dapi)
        print(f'loading {s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        labels = io.imread(f'{s_scene} nuclei{nuc_diam} - Nuclei Segmentation Basins.tif')
        print(f'loading {d_match[s_scene]}')
        cell_labels = io.imread(d_match[s_scene])
        #nuclear features
        df_feat = features.extract_feat(labels,dapi, properties=(['mean_intensity']))
        df_feat.columns = [f'{item}_segmented-nuclei' for item in df_feat.columns]
        df_feat.index = [f'{s_sample}_scene{s_scene.split("reg")[1]}_cell{item}' for item in df_feat.index]

        #get subcellular regions
        cyto = features.label_difference(labels,cell_labels)
        d_loc_nuc = features.subcellular_regions(labels, distance_short=2, distance_long=4)
        d_loc_cell = features.subcellular_regions(cell_labels, distance_short=2, distance_long=4)
        d_loc = {'nuclei':labels,'cell':cell_labels,'cytoplasm':cyto,
         'nucmem':d_loc_nuc['membrane'][0],'cellmem':d_loc_cell['membrane'][0],
         'perinuc4':d_loc_nuc['ring'][1],'exp4':d_loc_nuc['grown'][1],
         'nucadj2':d_loc_nuc['straddle'][0],'celladj2':d_loc_cell['straddle'][0]}
        #subdir organized by slide or scene
        if os.path.exists(f'{subdir}/{s_sample}'):
            os.chdir(f'{subdir}/{s_sample}')
        elif os.path.exists(f'{subdir}/{s_scene}'):
            os.chdir(f'{subdir}/{s_scene}')
        else:
            os.chdir(f'{subdir}')
        df_img = parse_img()
        df_img['round_int'] = [int(re.sub('[^0-9]','', item)) for item in df_img.rounds] 
        df_img = df_img[df_img.round_int < 90]
        df_img = df_img.sort_values('round_int')
        df_scene = df_img# one scene [df_img.scene==s_scene.split("-Scene-")[1].split("_")[0]]

        #load each image
        for s_index in df_scene.index:
                intensity_image = io.imread(s_index)
                df_thresh.loc[s_index,'threshold_li'] =  filters.threshold_li(intensity_image)
                if intensity_image.mean() > 0:
                    df_thresh.loc[s_index,'threshold_otsu'] = filters.threshold_otsu(intensity_image)
                    df_thresh.loc[s_index,'threshold_triangle'] = filters.threshold_triangle(intensity_image)
                s_marker = df_scene.loc[s_index,'marker']
                print(f'extracting features {s_marker}')
                #if s_marker == 'DAPI':
                #    s_marker = s_marker + f'{df_scene.loc[s_index,"rounds"].split("cyc")[1]}'
                for s_loc, a_loc in d_loc.items():
                    if s_loc == 'nuclei':
                        df_marker_loc = features.extract_feat(a_loc,intensity_image, properties=(['mean_intensity','centroid','area','eccentricity']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_centroid-0',f'{s_marker}_{s_loc}_centroid-1',f'{s_marker}_{s_loc}_area',f'{s_marker}_{s_loc}_eccentricity']
                    elif s_loc == 'cell':
                        df_marker_loc = features.extract_feat(a_loc,intensity_image, properties=(['mean_intensity','euler_number','area','eccentricity']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}',f'{s_marker}_{s_loc}_euler',f'{s_marker}_{s_loc}_area',f'{s_marker}_{s_loc}_eccentricity']
                    else:
                        df_marker_loc = features.extract_feat(a_loc,intensity_image, properties=(['mean_intensity']))
                        df_marker_loc.columns = [f'{s_marker}_{s_loc}']

                    #drop zero from array, set array ids as index
                    df_marker_loc.index = sorted(np.unique(a_loc)[1::])
                    df_marker_loc.index = [f'{s_sample}_scene{s_scene.split("reg")[1]}_cell{item}' for item in df_marker_loc.index]
                    df_feat = df_feat.merge(df_marker_loc, left_index=True,right_index=True,how='left',suffixes=('',f'{s_marker}_{s_loc}'))
        df_sample = df_sample.append(df_feat)
    return(df_sample, df_thresh)

def convert_tif(regdir,b_mkdir=True):
    '''
    convert codex tif to standard tif
    '''
    cwd = os.getcwd()
    os.chdir(regdir)
    for s_dir in sorted(os.listdir()):
        if s_dir.find('reg')== 0:
            os.chdir(s_dir)
            for s_file in sorted(os.listdir()):
                if s_file.find('.tif')>-1:
                    #s_round = s_file.split("Cycle(")[1].split(").ome.tif")[0]
                    #print(f'stain {s_round}')
                    #s_dir_new = s_dir.split('_')[2] + '-Scene-0' + s_dir.split('F-')[1]
                    #s_tissue_dir = s_dir.split('_F-')[0]
                    if b_mkdir:
                        preprocess.cmif_mkdir([f'{regdir}/converted_{s_dir}'])
                    a_dapi = skimage.io.imread(s_file)
                    with skimage.external.tifffile.TiffWriter(f'{regdir}/converted_{s_dir}/{s_file}') as tif:
                        tif.save(a_dapi)
            os.chdir('..')
    os.chdir(cwd)

def visualize_reg_images(s_sample,regdir,qcdir,color='ch001'):
    """
    array registered images to check tissue identity, focus, etc.
    """
    #check registration
    preprocess.cmif_mkdir([f'{qcdir}/RegisteredImages'])
    cwd = os.getcwd()
    os.chdir(regdir)
    #for idx, s_dir in enumerate(sorted(os.listdir())):
    #    os.chdir(s_dir)
    #    s_sample = s_dir.split('-Scene')[0]
    #    print(s_sample)
    df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='reg',s_split='_')
    df_img.rename({'data':'scene'},axis=1,inplace=True)
    df_img['slide'] = s_sample 
    df_img['rounds'] = [item[1] for item in [item.split('_') for item in df_img.index]]
    df_img['color'] = [item[2] for item in [item.split('_') for item in df_img.index]]
    df_img['marker'] = [item[3].split('.')[0] for item in [item.split('_') for item in df_img.index]]
    ls_scene = sorted(set(df_img.scene))
    for s_scene in ls_scene:
            print(s_scene)
            df_img_scene = df_img[df_img.scene == s_scene]
            df_img_stain = df_img_scene[df_img_scene.color==color]
            df_img_sort = df_img_stain.sort_values(['rounds'])
            i_sqrt = math.ceil(math.sqrt(len(df_img_sort)))
            fig = mpimage.array_img(df_img_sort,s_column='color',s_row='rounds',s_label='marker',tu_array=(i_sqrt,i_sqrt),tu_fig=(16,14))
            fig.savefig(f'{qcdir}/RegisteredImages/{s_scene}_registered_{color}.png')
    os.chdir(cwd)
    return(df_img_sort)

def rename_files(d_rename,dir,b_test=True):
    """
    change file names
    """
    cwd = os.getcwd()
    os.chdir(dir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        if s_dir.find('converted') == 0:
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
        if s_dir.find('converted') == 0:
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
        if s_dir.find('converted') == 0:
            s_path = f'{dir}/{s_dir}'
            os.chdir(s_path)
            #s_sample = s_dir.split('-Scene')[0]
            df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='R0',s_split='_')
            df_img.rename({'data':'rounds'},axis=1,inplace=True)
            df_img['scene'] = [item[1] for item in [item.split('_') for item in df_img.index]]
            df_img['color'] = [item[3] for item in [item.split('_') for item in df_img.index]]
            df_img['marker'] = [item[2].split('.')[0] for item in [item.split('_') for item in df_img.index]]
            print(s_dir)
            for key, dapi_item in dapi_copy.items():
                df_dapi = df_img[(df_img.rounds== key.split('_')[0]) & (df_img.color=='c1')]
                s_dapi = df_dapi.loc[:,'marker'][0]
                preprocess.copy_dapis(s_r_old=key,s_r_new=f'R{dapi_item}_',s_c_old='_c1_',
                 s_c_new='_c2_',s_find=f'_{s_dapi}_c1_ORG.tif',b_test=testbool,type=type)
            i_count=0
            for idx,(key, item) in enumerate(marker_copy.items()):
                preprocess.copy_markers(df_img, s_original=key, ls_copy = item,
                 i_last_round= dapi_item + i_count, b_test=testbool,type=type)
                i_count=i_count + len(item)
    return(df_img)

def segmentation_thresholds(regdir,qcdir, d_segment):
    """
    visualize binary mask of segmentaiton threholds
    need to edit
    """
    preprocess.cmif_mkdir([f'{qcdir}/Segmentation'])
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        if s_dir.find('converted') == 0:
            s_path = f'{regdir}/{s_dir}'
            os.chdir(s_path)
            df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='R',s_split='_')
            df_img.rename({'data':'rounds'},axis=1,inplace=True)
            df_img['scene'] = [item[1] for item in [item.split('_') for item in df_img.index]]
            df_img['color'] = [item[3] for item in [item.split('_') for item in df_img.index]]
            df_img['marker'] = [item[2].split('.')[0] for item in [item.split('_') for item in df_img.index]]
            s_sample = s_dir
            print(s_sample)
            d_seg = preprocess.check_seg_markers(df_img,d_segment, i_rows=1, t_figsize=(6,6)) #few scenes
            for key, fig in d_seg.items():
                fig.savefig(f'{qcdir}/Segmentation/{s_dir}_{key}_segmentation.png')
    return(df_img)

def parse_converted(dir):
    '''
    parse codex filenames (coverted)
    '''
    cwd = os.getcwd()
    os.chdir(dir)
    df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='R',s_split='_')
    df_img.rename({'data':'rounds'},axis=1,inplace=True)
    df_img['scene'] = [item[1] for item in [item.split('_') for item in df_img.index]]
    df_img['color'] = [item[3] for item in [item.split('_') for item in df_img.index]]
    df_img['marker'] = [item[2] for item in [item.split('_') for item in df_img.index]]
    os.chdir(cwd)
    return(df_img)

def segmentation_inputs(s_sample,regdir,segdir,d_segment,b_start=False):
    """
    make inputs for guillaumes segmentation
    """
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        if s_dir.find('convert')== 0:
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

def prepare_dataframe(s_sample,s_dapi,dapi_thresh,d_channel,ls_exclude,segdir,b_afsub=False):
    '''
    filter data by last dapi, standard location, subtract AF, output treshold csv
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
    df_dapi_mi = process.filter_dapi(df_mi,df_xy,s_dapi,dapi_thresh,b_images=True)
    df_t = process.load_meta(s_sample, s_path='./',type='LocationCsv')
    #filter mean intensity by biomarker location in metadata
    df_filter_mi = process.filter_loc(df_dapi_mi,df_t)
    df_filter_mi.to_csv(f'features_{s_sample}_FilteredMeanIntensity_{s_dapi}{dapi_thresh}.csv')
    if b_afsub:
        #load metadata
        df_t = pd.read_csv(f'metadata_{s_sample}_RoundsCyclesTableExposure.csv',index_col=0,header=0)
        #normalize by exposure time, and save to csv
        lb_columns = [len(set([item]).intersection(set(df_t.index)))>0 for item in [item.split('_')[0] for item in df_filter_mi.columns]]
        df_filter_mi = df_filter_mi.loc[:,lb_columns]
        df_norm = process.exposure_norm(df_filter_mi,df_t)     
        df_norm.to_csv(f'features_{s_sample}_ExpNormalizedMeanIntensity_{s_dapi}{dapi_thresh}.csv')
        #subtract AF channels in data
        df_sub,ls_sub,ls_record = process.af_subtract(df_norm,df_t,d_channel,ls_exclude)
        df_out = process.output_subtract(df_sub,df_t)
        df_out.to_csv(f'features_{s_sample}_AFSubtractedMeanIntensity_{s_dapi}{dapi_thresh}.csv')
        f = open(f"{s_sample}_AFsubtractionData.txt", "w")
        f.writelines(ls_record)
        f.close()
    else:
        df_out = df_filter_mi
    #output thresholding csv
    df_out = process.add_scene(df_out) #df_out
    df_thresh = process.make_thresh_df(df_out,ls_drop=None)
    df_thresh.to_csv(f'thresh_XX_{s_sample}.csv')

def multipage_tiff(d_combos,s_dapi,regdir):
    '''
    make custom overlays, either original of AF subtracted, save at 8 bit for size, and thresholding
    '''
    os.chdir(regdir)
    for idx, s_dir in enumerate(sorted(os.listdir())):
        if s_dir.find('convert')== 0:
            s_path = f'{regdir}/{s_dir}'
            os.chdir(s_path)
            df_img = mpimage.filename_dataframe(s_end = ".tif",s_start='R',s_split='_')
            df_img.rename({'data':'rounds'},axis=1,inplace=True)
            df_img['color'] = [item[3] for item in [item.split('_') for item in df_img.index]]
            df_img['marker'] = [item[2] for item in [item.split('_') for item in df_img.index]]
            df_img['scene'] = [item[1] for item in [item.split('_') for item in df_img.index]]
            df_img['imagetype'] = [item[4].split('.')[0] for item in [item.split('_') for item in df_img.index]]
            df_dapi =  df_img[df_img.marker.str.contains(s_dapi.split('_')[0])]
            df_img_stain = df_img[(~df_img.marker.str.contains('DAPI'))]
            #check
            es_test = set()
            for key, item in d_combos.items():
                es_test = es_test.union(item)
            print(set(df_img_stain.marker) - es_test)
            process.custom_overlays(d_combos, df_img_stain, df_dapi)
        else:
            continue
        
def load_crop_labels(d_crop,tu_dim,segdir,cropdir,s_find='Nuclei Segmentation Basins'):
    """
    crop the segmentation basins (cell of nuceli) to same coord as images for veiwing in Napari
    s_find: 'exp5_CellSegmentationBasins' or 'Nuclei Segmentation Basins'
    """
    cwd = os.getcwd()
    for s_scene, xy_cropcoor in d_crop.items():
        print(s_scene)
        s_sample = s_scene.split('-Scene-')[0]
        os.chdir(f'{segdir}')

        for s_file in os.listdir():
            if s_file.find(s_find) > -1: #Nuclei Segmentation Basins.tif #Cell Segmentation Basins.tif
                if s_file.find(s_scene.split(s_sample)[1]) > -1:
                    a_seg = skimage.io.imread(s_file)
                    a_crop = a_seg[(xy_cropcoor[1]):(xy_cropcoor[1]+tu_dim[1]),(xy_cropcoor[0]):(xy_cropcoor[0]+tu_dim[0])]
                    s_coor = f'x{xy_cropcoor[0]}y{xy_cropcoor[1]}.tif'
                    #crop file
                    s_file_new = f'{cropdir}/{s_sample}_{s_file.replace(" ","").replace(".tif",s_coor)}'
                    print(s_file_new)
                    skimage.io.imsave(s_file_new,a_crop)
    os.chdir(cwd)
