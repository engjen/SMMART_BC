####
# title: process.py
#
# language: Python3.6
# date: 2019-05-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 library to process cyclic data and images after segmentation
####

#libraries
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage
import copy
import re
import seaborn as sns
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

#function cellpose
def load_cellpose_df(ls_sample, segdir):
    '''
    load all full feature dataframes in sample list
    '''
    df_mi_full = pd.DataFrame()
    for idx, s_sample in enumerate(ls_sample):
        print(f'Loading features_{s_sample}_MeanIntensity_Centroid_Shape.csv')
        df_tt = pd.read_csv(f'{segdir}/features_{s_sample}_MeanIntensity_Centroid_Shape.csv',index_col=0)
        df_tt['slide'] = s_sample.split('-Scene')[0]
        df_tt['scene'] = [item.split('_')[1] for item in df_tt.index]
        df_mi_full = df_mi_full.append(df_tt,sort=True)
    #add scene
    df_mi_full['slide_scene'] = df_mi_full.slide + '_' + df_mi_full.scene
    print('')
    return(df_mi_full)

# load li thresholds
def load_li(ls_sample, s_thresh, man_thresh):
    '''
    load threshold on the segmentation marker images acquired during feature extraction
    '''
    df_img_all =pd.DataFrame()
    for s_sample in ls_sample:
        print(f'Loading thresh_{s_sample}_ThresholdLi.csv')
        df_img = pd.read_csv(f'thresh_{s_sample}_ThresholdLi.csv', index_col=0)
        df_img['rounds'] = [item.split('_')[0].split('Registered-')[1] for item in df_img.index]
        df_img['color'] = [item.split('_')[-2] for item in df_img.index]
        df_img['slide'] = [item.split('_')[2].split('-Scene-')[0] for item in df_img.index]
        df_img['scene'] = [item.split('_')[2].split('-Scene-')[1] for item in df_img.index]
        df_img['slide_scene'] = df_img.slide + '_scene' + df_img.scene
        #parse file name for biomarker
        for s_index in df_img.index:
            #print(s_index)
            s_color = df_img.loc[s_index,'color']
            if s_color == 'c1':
                s_marker = f"DAPI{df_img.loc[s_index,'rounds'].split('R')[1]}"
            elif s_color == 'c2':
                s_marker = s_index.split('_')[1].split('.')[0]
            elif s_color == 'c3':
                s_marker = s_index.split('_')[1].split('.')[1]
            elif s_color == 'c4':
                s_marker = s_index.split('_')[1].split('.')[2]
            elif s_color == 'c5':
                s_marker = s_index.split('_')[1].split('.')[3]
            else: print('Error')
            df_img.loc[s_index,'marker'] = s_marker
        df_img_all = df_img_all.append(df_img)
    print('')
    #manually override too low Ecad thresh
    if s_thresh !='':
        df_img_all.loc[df_img_all[(df_img_all.marker==s_thresh) & (df_img_all.threshold_li < man_thresh)].index, 'threshold_li'] = man_thresh
    return(df_img_all)

def filter_cellpose_xy(df_mi_full,ls_centroid = ['DAPI2_nuclei_area', 'DAPI2_nuclei_centroid-0', 'DAPI2_nuclei_centroid-1','DAPI2_nuclei_eccentricity']):
    '''
    select the nuclei centoids, area, eccentricity from a marker 
    default: use DAPI2
    '''
    #NOTE add area
    df_xy = df_mi_full.loc[:,ls_centroid]
    print('QC: make sure centroids dont have too many NAs')
    print(df_xy.isna().sum())
    print('')
    df_xy = df_xy.dropna(axis=0,how='any')
    df_xy.columns = ['nuclei_area','DAPI_Y','DAPI_X','nuclei_eccentricity']
    df_xy['slide_scene'] = [item.split('_cell')[0] for item in df_xy.index]
    return(df_xy)

def drop_last_rounds(df_img_all,ls_filter,df_mi_full):
    '''
    drop any rounds after the last round DAPI filter 
    '''
    df_img_all['round_ord'] = [re.sub('Q','.5', item) for item in df_img_all.rounds] 
    df_img_all['round_ord'] = [float(re.sub('[^0-9.]','', item)) for item in df_img_all.round_ord]
    i_max = df_img_all[df_img_all.marker.isin([item.split('_')[0] for item in ls_filter])].sort_values('round_ord').iloc[-1].round_ord
    print(f'Dropping markers after round {i_max}')
    ls_drop_marker = [item + '_' for item in sorted(set(df_img_all[(df_img_all.round_ord>i_max)].marker))]
    [print(item) for item in ls_drop_marker]
    print('')
    [df_mi_full.drop(df_mi_full.columns[df_mi_full.columns.str.contains(item)],axis=1,inplace=True) for item in ls_drop_marker]
    return(df_mi_full,i_max)

def plot_thresh(df_img_all,s_thresh):
    '''
    tissues: plot threshold across all tissues
    (negative scenes will drive down the mean
    '''
    ls_slides = sorted(set(df_mi_full.slide))
    df_plot = df_img_all[(df_img_all.marker==s_thresh)].loc[:,['threshold_li']]
    fig,ax=plt.subplots(figsize=(4,3.5))
    sns.stripplot(data=df_plot)
    sns.barplot(data=df_plot, alpha=0.5)
    labels = ax.get_xticklabels
    plt.tight_layout()
    fig.savefig(f'{qcdir}/QC_EcadThresh_{".".join(ls_slides)}.png')

def fill_cellpose_nas(df_mi_full,ls_marker_cyto,s_thresh='Ecad',man_thresh=1000): 
    '''
    some nuclei don't have a cytoplasm, replace NA with perinuc5
    '''
    df = df_mi_full.copy(deep=True)
    # since segmentation was run on ecad, use ecad threshold
    print(f'Finding {s_thresh} positive cells')
    ls_neg_cells = (df_mi_full[~(df_mi_full.loc[:,f'{s_thresh}_cytoplasm'] > man_thresh)]).index.tolist()#= ls_neg_cells + ls_neg_slide
    print('')
    # replace cells without cytoplasm (ecad) with perinuc 5
    print(f'For cells that are {s_thresh} negative:')
    for s_marker in ls_marker_cyto:
        print(f'Replace  {s_marker}_cytoplasm nas')
        df.loc[ls_neg_cells,f'{s_marker}_cytoplasm'] = df.loc[ls_neg_cells,f'{s_marker}_perinuc5'] 
        print(f'with {s_marker}_perinuc5')
    df[f'{s_thresh}_negative'] = df.index.isin(ls_neg_cells)
    return(df)

def shrink_seg_regions(df_mi_full,s_thresh,ls_celline=[],ls_shrunk=[]):
    '''
    For markers with stromal to tumor bleedthrough, use shrunken segmentation region
    '''
    #enforce cell lines as tumor
    print('')
    if len(ls_celline) > 0:
        print([f'Enforce {item} as tumor' for item in ls_celline])
    ls_ecad_cells = df_mi_full[~df_mi_full.loc[:,f'{s_thresh}_negative']].index
    ls_tumor_cells = (df_mi_full[(df_mi_full.index.isin(ls_ecad_cells)) | (df_mi_full.slide_scene.isin(ls_celline))]).index
    ls_stromal_cells = (df_mi_full[~df_mi_full.index.isin(ls_tumor_cells)]).index
    #relplace tumor cell CD44 and Vim with shrunken area (only helps bleed trough a little)
    print('For markers with stromal to tumor bleedthrough, use shrunken segmentation region:')
    for s_marker in ls_shrunk:
        print(f'Replace  {s_marker.split("_")[0]}_perinuc5 in tumor cells with')
        df_mi_full.loc[ls_tumor_cells,f'{s_marker.split("_")[0]}_perinuc5'] = df_mi_full.loc[ls_tumor_cells,f'{s_marker}']
        print(f'with {s_marker}')
    print('')
    return(df_mi_full)

def fill_membrane_nas(df_mi_full, df_mi_mem,s_thresh='Ecad',ls_membrane=['HER2']):
    '''
    fill cell membrane nsa with expanded nuclei nas
    '''
    ls_neg = df_mi_full[(df_mi_full.loc[:,f'{s_thresh}_negative']) & (df_mi_full.index.isin(df_mi_mem.index))].index
    ls_pos = df_mi_full[(~df_mi_full.loc[:,f'{s_thresh}_negative']) & (df_mi_full.index.isin(df_mi_mem.index))].index
    for s_membrane in ls_membrane:
        print(f'Replace {s_membrane}_cellmem25 nas \n with {s_membrane}_exp5nucmembrane25')
        df_mi_mem.loc[ls_neg,f'{s_membrane}_cellmem25'] =  df_mi_mem.loc[ls_neg,f'{s_membrane}_exp5nucmembrane25']
        ls_na = df_mi_mem.loc[df_mi_mem.loc[:,f'{s_membrane}_cellmem25'].isna(),:].index
        df_mi_mem.loc[ls_na,f'{s_membrane}_cellmem25'] = df_mi_mem.loc[ls_na,f'{s_membrane}_exp5nucmembrane25']
    df_merge = df_mi_full.merge(df_mi_mem, left_index=True, right_index=True)
    print('')
    return(df_merge)

def fill_bright_nas(ls_membrane,s_sample,s_thresh,df_mi_filled,segdir):
    if len(ls_membrane) > 0:
        print(f'Loading features_{s_sample}_BrightMeanIntensity.csv')
        df_mi_mem = pd.read_csv(f'{segdir}/features_{s_sample}_BrightMeanIntensity.csv',index_col=0)
        df_mi_mem_fill = fill_membrane_nas(df_mi_filled, df_mi_mem,s_thresh=s_thresh,ls_membrane=ls_membrane)
    else:
        df_mi_mem_fill = df_mi_filled
    return(df_mi_mem_fill)

def auto_threshold(df_mi,df_img_all):
    # # Auto threshold

    #make positive dataframe to check threhsolds 
    ls_scene = sorted(set(df_mi.slide_scene))

    df_pos_auto = pd.DataFrame()
    d_thresh_record= {}

    for s_slide_scene in ls_scene:
        print(f'Thresholding {s_slide_scene}')
        ls_index = df_mi[df_mi.slide_scene==s_slide_scene].index
        df_scene = pd.DataFrame(index=ls_index)
        df_img_scene = df_img_all[df_img_all.slide_scene==s_slide_scene]

        for s_index in df_img_scene.index:
            s_scene =f"{df_img_all.loc[s_index,'slide']}_scene{df_img_all.loc[s_index,'scene']}"
            s_marker = df_img_all.loc[s_index,'marker']
            s_columns = df_mi.columns[df_mi.columns.str.contains(f"{s_marker}_")]
            if len(s_columns)==1:
                s_marker_loc = s_columns[0]
            else:
                continue
            i_thresh = df_img_all.loc[s_index,'threshold_li']
            d_thresh_record.update({f'{s_scene}_{s_marker}':i_thresh})
            df_scene.loc[ls_index,s_marker_loc] = df_mi.loc[ls_index,s_marker_loc] >= i_thresh
        df_pos_auto = df_pos_auto.append(df_scene)
    return(df_pos_auto,d_thresh_record)

def positive_scatterplots(df_pos_auto,d_thresh_record,df_xy,ls_color,qcdir='.'):
    '''
    for marker in ls_color, plot positive cells location in tissue
    '''
    ls_scene = sorted(set(df_xy.slide_scene))

    for s_scene in ls_scene:
        print(f'Plotting {s_scene}')
        #negative cells = all cells even before dapi filtering
        df_neg = df_xy[(df_xy.slide_scene==s_scene)]
        #plot
        fig, ax = plt.subplots(2, ((len(ls_color))+1)//2, figsize=(18,12)) #figsize=(18,12)
        ax = ax.ravel()
        for ax_num, s_color in enumerate(ls_color):
            s_marker = s_color.split('_')[0]
            s_min = d_thresh_record[f"{s_scene}_{s_marker}"]
            #positive cells = positive cells based on threshold
            ls_pos_index = (df_pos_auto[df_pos_auto.loc[:,s_color]]).index
            df_color_pos = df_neg[df_neg.index.isin(ls_pos_index)]
            if len(df_color_pos)>=1:
                #plot negative cells
                ax[ax_num].scatter(data=df_neg,x='DAPI_X',y='DAPI_Y',color='silver',s=1)
                #plot positive cells
                ax[ax_num].scatter(data=df_color_pos, x='DAPI_X',y='DAPI_Y',color='DarkBlue',s=.5)
                      
                ax[ax_num].axis('equal')
                ax[ax_num].set_ylim(ax[ax_num].get_ylim()[::-1])
                ax[ax_num].set_title(f'{s_marker} min={int(s_min)}')
            else:
                ax[ax_num].set_title(f'{s_marker} min={int(s_min)}')
        ls_save = [item.split('_')[0] for item in ls_color]
        fig.suptitle(s_scene)
        fig.savefig(f'{qcdir}/QC_{".".join(ls_save)}_{s_scene}_auto.png')

def plot_thresh_results(df_img_all,df_pos_auto,d_thresh_record,df_xy,i_max,s_thresh,qcdir):
    ls_color = [item + '_nuclei' for item in df_img_all[(df_img_all.round_ord<=i_max) & (df_img_all.slide_scene==df_img_all.slide_scene.unique()[0]) & (df_img_all.marker.str.contains('DAPI'))].marker.tolist()]
    positive_scatterplots(df_pos_auto,d_thresh_record,df_xy,ls_color + [f'{s_thresh}_cytoplasm'],qcdir)
    return(ls_color)

def filter_dapi_cellpose(df_pos_auto,ls_color,df_mi,ls_filter,qcdir='.'):
    '''
    filter by cell positive for DAPI autotresholding, in rounds specified in ls_filter
    error
    '''
    #plot dapi thresholds
    df_pos_auto['slide_scene'] = [item.split('_cell')[0] for item in df_pos_auto.index]
    fig,ax=plt.subplots(figsize=(10,5))
    df_plot = df_pos_auto.loc[:,ls_color+['slide_scene']]
    df_scenes = df_plot.groupby('slide_scene').sum().T/df_plot.groupby('slide_scene').sum().max(axis=1)
    df_scenes.plot(ax=ax,colormap='tab20')
    ax.set_xticks(np.arange(0,(len(df_scenes.index)),1)) #+1
    ax.set_xticklabels([item.split('_')[0] for item in df_scenes.index])
    ax.set_ylim(0.5,1.1)
    ax.legend(loc=3)
    plt.tight_layout()
    df_pos_auto['slide'] = [item.split('_')[0] for item in df_pos_auto.index]
    ls_slides = sorted(set(df_pos_auto.slide))
    fig.savefig(f'{qcdir}/QC_DAPIRounds_lineplot_{".".join(ls_slides)}.png')
    #filter by first and last round dapi
    ls_dapi_index = df_pos_auto[df_pos_auto.loc[:,ls_filter].all(axis=1)].index
    #also filter by any dapi less than 1 in mean intensity
    ls_dapi_missing = df_mi[(df_mi.loc[:,ls_color] < 1).sum(axis=1) > 0].index.tolist()
    es_dapi_index = set(ls_dapi_index) - set(ls_dapi_missing)
    print(f'number of cells before DAPI filter = {len(df_mi)}')
    df_mi_filter = df_mi.loc[df_mi.index.isin(es_dapi_index),:]
    [print(f'filtering by {item}') for item in ls_filter]
    print(f'number of cells after DAPI filter = {len(df_mi_filter)}')
    #drop cells with euler numer > 1
    #
    #
    return(df_mi_filter)

def load_li_thresh(ls_sample, segdir):
    # load li thresholds
    os.chdir(segdir)
    df_img_all =pd.DataFrame()
    for s_sample in ls_sample:
        df_img = pd.read_csv(f'thresh_{s_sample}_ThresholdLi.csv', index_col=0)
        df_img['rounds'] = [item.split('_')[0].split('Registered-')[1] for item in df_img.index]
        df_img['color'] = [item.split('_')[-2] for item in df_img.index]
        df_img['slide'] = [item.split('_')[2].split('-Scene-')[0] for item in df_img.index]
        df_img['scene'] = [item.split('_')[2].split('-Scene-')[1] for item in df_img.index]
        df_img['slide_scene'] = df_img.slide + '_scene' + df_img.scene
        #parse file name for biomarker
        for s_index in df_img.index:
            #print(s_index)
            s_color = df_img.loc[s_index,'color']
            if s_color == 'c1':
                s_marker = f"DAPI{df_img.loc[s_index,'rounds'].split('R')[1]}"
            elif s_color == 'c2':
                s_marker = s_index.split('_')[1].split('.')[0]
            elif s_color == 'c3':
                s_marker = s_index.split('_')[1].split('.')[1]
            elif s_color == 'c4':
                s_marker = s_index.split('_')[1].split('.')[2]
            elif s_color == 'c5':
                s_marker = s_index.split('_')[1].split('.')[3]
            else: print('Error')
            df_img.loc[s_index,'marker'] = s_marker
        df_img_all = df_img_all.append(df_img)
    return(df_img_all)

def filter_standard(df_mi,d_channel,s_dapi):
    """
    If biomarkers have standard names according to preprocess.check_names,
    use the hard coded locations, adds any channels needed for af subtraction
    Input:
    df_mi= mean intensity dataframe 
    d_channel = dictionary of channel:background marker
    """
    es_standard = {'PDL1_Ring','pERK_Nuclei','CK19_Ring','pHH3_Nuclei','CK14_Ring','Ki67_Nuclei','Ki67r_Nuclei','Ecad_Ring','PCNA_Nuclei','HER2_Ring','ER_Nuclei','CD44_Ring',
        'aSMA_Ring','AR_Nuclei','pAKT_Ring','LamAC_Nuclei','CK5_Ring','EGFR_Ring','pRb_Nuclei','FoxP3_Nuclei','CK7_Ring','PDPN_Ring','CD4_Ring','PgR_Nuclei','Vim_Ring',
        'CD8_Ring','CD31_Ring','CD45_Ring','panCK_Ring','CD68_Ring','PD1_Ring','CD20_Ring','CK8_Ring','cPARP_Nuclei','ColIV_Ring','ColI_Ring','CK17_Ring',
        'H3K4_Nuclei','gH2AX_Nuclei','CD3_Ring','H3K27_Nuclei','53BP1_Nuclei','BCL2_Ring','GRNZB_Nuclei','LamB1_Nuclei','pS6RP_Ring','BAX_Nuclei','RAD51_Nuclei',
        'Glut1_Ring','CoxIV_Ring','LamB2_Nuclei','S100_Ring','BMP4_Ring','PgRc4_Nuclei','pRB_Nuclei','p63_Nuclei','p63_Ring','CGA_Ring','SYP_Ring','pS62MYC_Nuclei', 'HIF1a_Nuclei',
        'PDGFRa_Ring', 'BMP2_Ring','PgRb_Nuclei','MUC1_Ring','CSF1R_Ring','CAV1_Ring','CCND1_Nuclei','CC3_Nuclei' } #PgRb is second PgR in dataset
    #generate list of background markers needed for subtraction
    lls_d_channel = []
    for s_key,ls_item in d_channel.items():
        lls_d_channel = lls_d_channel + [ls_item]
    ls_background = []
    for ls_channel in lls_d_channel:
        ls_background = ls_background + [f'{ls_channel[0]}_Ring']
        ls_background = ls_background + [f'{ls_channel[1]}_Nuclei']
    #ls_background.append(f'{s_dapi}_Nuclei')
    ls_background.append(f'{s_dapi}')
    se_background = set(ls_background)
    es_common = set(df_mi.columns.tolist()).intersection(es_standard) | se_background
    df_filtered_mi = df_mi.loc[:,sorted(es_common)]
    return(df_filtered_mi, es_standard)

def filter_loc_cellpose(df_mi_filled, ls_marker_cyto, ls_custom,filter_na=True):
    '''
    get nuclei, perinuclei or cytoplasm, based on filter standard function
    '''
    __ , es_standard = filter_standard(pd.DataFrame(columns=['filter_standard']),{},'filter_standard')
    ls_marker = sorted(set([item.split('_')[0] for item in df_mi_filled.columns[(df_mi_filled.dtypes=='float64') & (~df_mi_filled.columns.str.contains('25'))]]))
    if ls_marker.count('mean') != 0:
        ls_marker.remove('mean')
    es_marker = set(ls_marker)
    se_stand = pd.Series(index=es_standard)
    es_dapi = set([item.split('_')[0] for item in df_mi_filled.columns[df_mi_filled.columns.str.contains('DAPI')]])
    es_nuc = set([item.split('_')[0] for item in se_stand[se_stand.index.str.contains('_Nuclei')].index])
    es_nuc_select = es_nuc.intersection(es_marker)
    print('Nuclear markers:')
    print(es_nuc_select)
    es_ring = set([item.split('_')[0] for item in se_stand[se_stand.index.str.contains('_Ring')].index])
    es_ring_select = es_ring.intersection(es_marker)
    es_cyto = set(ls_marker_cyto) #set([item.split('_')[0] for item in ls_marker_cyto])
    es_ring_only = es_ring_select - es_cyto
    print('Ring markers:')
    print(es_ring_only)
    print('Cytoplasm markers:')
    print(es_cyto)
    es_cust = set([item.split('_')[0] for item in ls_custom])
    es_left = es_marker - es_ring_only - es_cyto - es_nuc_select - es_dapi - es_cust
    print('Custom markers:')
    print(es_cust)
    print('Markers with Nuclei or Cyto not specified: take both nuclei and ring')
    print(es_left)
    ls_n = [item + '_nuclei' for item in sorted(es_left | es_nuc_select | es_dapi)]
    ls_pn = [item + '_perinuc5' for item in sorted(es_left | es_ring_only)] 
    ls_cyto = [item + '_cytoplasm' for item in sorted(es_cyto)]
    ls_all = ls_custom + ls_pn + ls_cyto + ls_n + ['slide_scene']
    print(f'Missing {set(ls_all) - set(df_mi_filled.columns)}') 
    df_filter = df_mi_filled.loc[:,ls_all]
    print('')
    if filter_na:
        print(f' NAs filtered: {len(df_filter) - len(df_filter.dropna())}')
        df_filter = df_filter.dropna()
    print('')
    return(df_filter,es_standard)

def marker_table(df_img_all,qcdir):
    '''
    make a nice rounds/channels/markers table
    '''
    df_img_all['round_ord'] = [re.sub('Q','.5', item) for item in df_img_all.rounds]
    df_img_all['round_ord'] = [re.sub('r','.25', item) for item in df_img_all.round_ord]
    df_img_all['round'] = [float(re.sub('[^0-9.]','', item)) for item in df_img_all.round_ord]
    df_marker = df_img_all[(df_img_all.slide_scene==df_img_all.slide_scene.unique()[0])].loc[:,['marker','round','color']].pivot('round','color')
    df_marker.index.name = None
    df_marker.to_csv(f'{qcdir}/MarkerTable.csv',header=None)

def filter_cellpose_df(s_sample,segdir,qcdir,s_thresh,ls_membrane,ls_marker_cyto,ls_custom,ls_filter,ls_shrunk,man_thresh = 900):
    '''
    go from full dataframe and membrane dataframe to filtered datframe and xy coordinate dataframe
    s_thresh='Ecad'
    ls_membrane = ['HER2']
    ls_marker_cyto = ['CK14','CK5','CK17','CK19','CK7','CK8','Ecad','HER2','EGFR']
    ls_custom = ['HER2_cellmem25']
    ls_filter = ['DAPI9_nuclei','DAPI2_nuclei']
    ls_shrunk = ['CD44_nucadj2','Vim_nucadj2']
    man_thresh = 900
    '''
    # new
    os.chdir(segdir)
    df_img_all = load_li([s_sample],s_thresh, man_thresh)
    df_mi_full = load_cellpose_df([s_sample], segdir)
    df_xy = filter_cellpose_xy(df_mi_full)
    df_mi_full, i_max = drop_last_rounds(df_img_all,ls_filter,df_mi_full)
    df_mi_filled = fill_cellpose_nas(df_mi_full,ls_marker_cyto,s_thresh=s_thresh,man_thresh=man_thresh)
    df_mi_filled = shrink_seg_regions(df_mi_filled,s_thresh,ls_celline=[],ls_shrunk=ls_shrunk)
    df_mi_mem_fill = fill_bright_nas(ls_membrane,s_sample,s_thresh,df_mi_filled,segdir)
    df_mi,es_standard = filter_loc_cellpose(df_mi_mem_fill, ls_marker_cyto, ls_custom)
    df_pos_auto,d_thresh_record = auto_threshold(df_mi,df_img_all)
    ls_color = plot_thresh_results(df_img_all,df_pos_auto,d_thresh_record,df_xy,i_max,s_thresh,qcdir)
    df_mi_filter = filter_dapi_cellpose(df_pos_auto,ls_color,df_mi,ls_filter,qcdir)
    df_mi_filter.to_csv(f'{segdir}/features_{s_sample}_FilteredMeanIntensity_{"_".join([item.split("_")[0] for item in ls_filter])}.csv')
    df_xy.to_csv(f'{segdir}/features_{s_sample}_CentroidXY.csv')
    return(df_mi_mem_fill,df_img_all)

def filter_cellpose_background(df_mi_filled, es_standard):
    '''
    given a set of standard biomarker subcellular locations, obtain the opposite subcellular location 
    and the mean intensity 
    input: df_mi = mean intensity dataframe with all biomarker locations
    es_standard = biomarker ring or nuclei 
    return: dataframe with each scene and the quantiles of the negative cells scene
    '''
    ls_rim = [item.replace('Nuclei','cytoplasm') for item in sorted(es_standard)]
    ls_nuc_ring = [item.replace('Ring','nuclei') for item in ls_rim]
    ls_nuc_ring.append('slide_scene')
    ls_nuc_ring = sorted(set(df_mi_filled.columns).intersection(set(ls_nuc_ring)))
    #quntiles
    df_bg =  df_mi_filled.loc[:,ls_nuc_ring].groupby('slide_scene').quantile(0) 
    df_bg.columns = [f'{item}' for item in df_bg.columns]
    for q in np.arange(0,1,.1):
        df_quantile = df_mi_filled.loc[:,ls_nuc_ring].groupby('slide_scene').quantile(q)
        df_bg = df_bg.merge(df_quantile,left_index=True, right_index=True, suffixes=('',f'_{str(int(q*10))}'))
    #drop duplicate
    ls_nuc_ring.remove('slide_scene')
    df_bg = df_bg.loc[:,~df_bg.columns.isin(ls_nuc_ring)]
    return(df_bg)

def filter_cellpose_df_old(df_mi_full):
    '''
    old
    '''
    #filter
    ls_select = [
        #nuclei
        'DAPI1_nuclei', 'DAPI2_nuclei', 'DAPI3_nuclei', 'DAPI4_nuclei','DAPI5_nuclei', 'DAPI5Q_nuclei',
        'DAPI6_nuclei', 'DAPI7_nuclei','DAPI8_nuclei', 'DAPI9_nuclei',
        'DAPI10_nuclei', 'DAPI11_nuclei','DAPI12_nuclei','DAPI12Q_nuclei',
        'ER_nuclei','AR_nuclei','PgR_nuclei',
        'Ki67_nuclei',  'pRB_nuclei','PCNA_nuclei', 'pHH3_nuclei',
        'FoxP3_nuclei', 'GRNZB_nuclei',
        'H3K27_nuclei', 'H3K4_nuclei',  
        'LamAC_nuclei', 'LamB1_nuclei', 'LamB2_nuclei',
        'HIF1a_nuclei', 'pERK_nuclei', 'cPARP_nuclei', 'gH2AX_nuclei',

        #perinuc5     
        'CD44_perinuc5',
        'CD20_perinuc5', 'CD31_perinuc5',
        'CD3_perinuc5', 'CD45_perinuc5', 'CD4_perinuc5',
        'CD68_perinuc5', 'CD8_perinuc5','pS6RP_perinuc5',
        'ColIV_perinuc5', 'ColI_perinuc5', 'CoxIV_perinuc5',
        'PD1_perinuc5', 'PDPN_perinuc5','PDGFRa_perinuc5', 
        'Vim_perinuc5', 'aSMA_perinuc5','BMP2_perinuc5',
        #cytoplasm
        #'pAKT_cytoplasm','Glut1_cytoplasm',
        'CK14_cytoplasm','CK5_cytoplasm','CK17_cytoplasm',
        'CK19_cytoplasm','CK7_cytoplasm','CK8_cytoplasm',
        'Ecad_cytoplasm','HER2_cytoplasm','EGFR_cytoplasm',
        #other
        'slide_scene',
        #'area_segmented-nuclei', #'area_segmented-cells',
        #'eccentricity_segmented-nuclei', #'eccentricity_segmented-cells',
        #'mean_intensity_segmented-nuclei', #'mean_intensity_segmented-cells',
        ]

    ls_negative = df_mi_full.columns[df_mi_full.columns.str.contains('_negative')].tolist()
    #print(type(ls_negative))
    ls_select = ls_select + ls_negative

    df_mi_nas = df_mi_full.loc[:,df_mi_full.columns.isin(ls_select)]
    print(f'Selected makers that were missing from mean intensity {set(ls_select) - set(df_mi_nas.columns)}')
    #fiter out nas
    print(f'Number on df_mi nas = {df_mi_nas.isna().sum().max()}')
    df_mi = df_mi_nas.dropna(axis=0,how='any')
    return(df_mi,df_mi_nas)

###### below: functions for guillaumes features  ########

def load_mi(s_sample, s_path='./', b_set_index=True):
    """
    input:
        s_sample: string with sample name
        s_path: file path to data, default is current folder
        b_set_index: 

    output:
        df_mi: dateframe with mean intensity
          each row is a cell, each column is a biomarker_location

    description:
      load the mean intensity dataframe
    """
    print(f'features_{s_sample}_MeanIntensity.tsv')
    df_mi = pd.read_csv(
        f'{s_path}features_{s_sample}_MeanIntensity.tsv',
        sep='\t',
        index_col=0
        )
    if b_set_index:
        df_mi = df_mi.set_index(f'{s_sample}_' + df_mi.index.astype(str))
    return(df_mi)

def load_xy(s_sample, s_path='./', b_set_index=True):
    """
    input:
        s_sample: string with sample name
        s_path: file path to data, default is current folder
        b_set_index: 

    output:
        df_mi: dateframe with mean intensity
          each row is a cell, each column is a biomarker_location

    description:
      load the mean intensity dataframe
    """
    print(f'features_{s_sample}_CentroidY.tsv')
    df_y = pd.read_csv(
        f'features_{s_sample}_CentroidY.tsv',
        sep='\t',
        index_col=0
        )
    if b_set_index:
        df_y = df_y.set_index(f'{s_sample}_' + df_y.index.astype(str))

    print(f'features_{s_sample}_CentroidX.tsv')
    df_x = pd.read_csv(
        f'features_{s_sample}_CentroidX.tsv',
        sep='\t',
        index_col=0
        )
    if b_set_index:
        df_x = df_x.set_index(f'{s_sample}_' + df_x.index.astype(str))
    #merge the x and y dataframes
    df_xy = pd.merge(df_x,df_y,left_index=True,right_index=True,suffixes=('_X', '_Y'))
    return(df_xy)

def add_scene(df,i_scene_index=1,s_group='scene'):
    """
    decription: add a coulmn with a grouping to dataframe that has grouping in the index
    """
    lst = df.index.str.split('_')
    lst2 = [item[i_scene_index] for item in lst]
    df[s_group] = lst2
    return(df)

def filter_dapi(df_mi,df_xy,s_dapi='DAPI11_Nuclei',dapi_thresh=1000,b_images=False,t_figsize=(8,8)):
    """
    description: return a dataframe where all cells have DAPI brigter than a threshold
    right now the plotting works!
    """
    df_filtered_mi = df_mi.copy(deep=True)
    #get tissue id from the dataframe
    s_tissue = df_mi.index[0].split('_')[0]
    #DAPI filter
    df_filtered_mi = df_filtered_mi[df_filtered_mi.loc[:,s_dapi]>dapi_thresh]
    print(f'Cells before DAPI filter = {len(df_mi)}')
    print(f'Cells after DAPI filter = {len(df_filtered_mi)}')
    df_filtered_mi.index.name='UNIQID'
    if b_images:
        ls_scene=list(set(df_xy.scene))
        ls_scene.sort()
        for s_scene in ls_scene:
            df_pos = df_xy.loc[df_filtered_mi.index.tolist()]
            df_pos_scene = df_pos[df_pos.scene==s_scene]
            if len(df_pos_scene) >= 1:
                fig,ax=plt.subplots(figsize=t_figsize)
                ax.scatter(x=df_xy[df_xy.scene==s_scene].loc[:,'DAPI_X'], y=df_xy[df_xy.scene==s_scene].loc[:,'DAPI_Y'], color='silver',label='DAPI neg', s=2)
                ax.scatter(x=df_pos_scene.loc[:,'DAPI_X'], y=df_pos_scene.loc[:,'DAPI_Y'], color='DarkBlue',label='DAPI pos',s=2)
                ax.axis('equal')
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.set_title(f'{s_scene}_DAPI')
                plt.legend(markerscale=3)
                fig.savefig(f'{s_tissue}_{s_scene}_{s_dapi}{dapi_thresh}.png')
    return(df_filtered_mi)

def load_meta(s_sample, s_path='./',type='csv'):
    """
    load rounds cycles table
    make sure to specify location for use with downstream functions
    make sure to add rows for any biomarkers used for analysis or processing
    """
    #tab or space delimited
    if type == 'Location':
        print(f'metadata_{s_sample}_RoundsCyclesTable_location.txt')
        df_t = pd.read_csv(
            f'metadata_{s_sample}_RoundsCyclesTable_location.txt',
            delim_whitespace=True,
            header=None,
            index_col=False,
            names=['marker', 'rounds','color','minimum', 'maximum', 'exposure', 'refexp','location'],
            )
        df_t = df_t.set_index(f'{s_sample}_' + df_t.index.astype(str))
        df_t.replace({'Nucleus':'Nuclei'},inplace=True)
        df_t['marker_loc'] = df_t.marker + '_' + df_t.location
        df_t.set_index(keys='marker_loc',inplace=True)
    elif type == 'csv':
        print(f'metadata_{s_sample}_RoundsCyclesTable.csv')
        df_t = pd.read_csv(
            f'metadata_{s_sample}_RoundsCyclesTable.csv',
            header=0,
            index_col=0,
            names=['rounds','color','minimum', 'maximum', 'exposure', 'refexp','location'],#'marker',
            )
        #df_t = df_t.set_index(f'{s_sample}_' + df_t.index.astype(str))
        df_t.replace({'Nucleus':'Nuclei'},inplace=True)
    #
    elif type == 'LocationCsv':
        print(f'metadata_{s_sample}_RoundsCyclesTable_location.csv')
        df_t = pd.read_csv(
            f'metadata_{s_sample}_RoundsCyclesTable_location.csv',
            header=0,
            index_col=False,
            names=['marker', 'rounds','color','minimum', 'maximum', 'exposure', 'refexp','location'],
            )
        df_t = df_t.set_index(f'{s_sample}_' + df_t.index.astype(str))
        df_t.replace({'Nucleus':'Nuclei'},inplace=True)
        df_t['marker_loc'] = df_t.marker + '_' + df_t.location
        df_t.set_index(keys='marker_loc',inplace=True)
    else:
        print(f'metadata_{s_sample}_RoundsCyclesTable.txt')
        df_t = pd.read_csv(
            f'metadata_{s_sample}_RoundsCyclesTable.txt',
            delim_whitespace=True,
            header=None,
            index_col=False,
            names=['rounds','color','minimum', 'maximum', 'exposure', 'refexp','location'],#'marker',
            )
        df_t = df_t.set_index(f'{s_sample}_' + df_t.index.astype(str))
        df_t.replace({'Nucleus':'Nuclei'},inplace=True)
    return(df_t)

def add_exposure_roundscyles(df_tc, df_expc,es_standard,ls_dapi = ['DAPI12_Nuclei']):
    """
    df_exp = dataframe of exposure times with columns [0, 1,2,3,4]
            and index with czi image names
    df_t = metadata with dataframe with ['marker','exposure']
    """
    df_t = copy.copy(df_tc)
    df_exp = copy.copy(df_expc)
    df_t['location'] = ''
    df_t.drop([item.split('_')[0] for item in ls_dapi], inplace=True)
    df_exp.columns = ['c' + str(int(item)+1) for item in df_exp.columns]
    df_exp['rounds'] = [item.split('_')[0] for item in df_exp.index]
    for s_index in df_t.index:
        s_channel = df_t.loc[s_index,'colors']
        s_round = df_t.loc[s_index, 'rounds']
        print(s_round)
        #look up exposure time for marker in metadata
        df_t_image = df_exp[(df_exp.rounds==s_round)]
        if len(df_t_image) > 0:
                i_exposure = df_t_image.loc[:,s_channel]
                df_t.loc[s_index,'exposure'] = i_exposure[0]
                df_t.loc[s_index,'refexp'] = i_exposure[0]
        else:
                print(f'{s_marker} has no recorded exposure time')
        s_ring = s_index + '_Ring'
        s_nuc = s_index + '_Nuclei'
        ls_loc = sorted(es_standard.intersection({s_ring,s_nuc}))
        if len(ls_loc) == 1:
            df_t.loc[s_index,'location'] = ls_loc[0].split('_')[1]
    return(df_t)

def filter_loc(df_mi,df_t):
    """
    filters columns of dataframe based on locations selected in metadata_location table
    """
    ls_bio_loc = df_t.index.tolist()
    df_filtered_mi = df_mi.loc[:,ls_bio_loc]
    return(df_filtered_mi)

#R0c2 R0c3 R0c4 R0c5 panCK CK14  Ki67 CK19  R1rc2 R1rc3 Ki67r R1rc5 PCNA HER2 ER Ecad aSMA AR pAKT
#CD44 CK5 EGFR pRB LamAC pHH3 PDPN pERK FoxP3 R5Qc2 R5Qc3 R5Qc4 R5Qc5 CK7 CD68 PD1 CD45 Vim CD8 CD4 PgR CK8 cPARP ColIV CD20 CK17
#H3K4 gH2AX ColI H3K27 pS6RP CD31 GRNZB LamB1 CoxIV HIF1a CD3 Glut1 PDGFRa LamB2 BMP2 R12Qc2 R12Qc3 R12Qc4 R12Qc5 DAPI12

def filter_background(df_mi, es_standard):
    '''
    given a set of standard biomarker subcellular locations, obtain the opposite subcellular location 
    and the mean intensity 
    input: df_mi = mean intensity dataframe with all biomarker locations
    es_standard = biomarker ring or nuclei 
    return: dataframe with each scene and the quantiles of the negative cells
    '''
    ls_rim = [item.replace('Nuclei','Rim') for item in sorted(es_standard)]
    ls_nuc_rim =  [item.replace('Ring','Nuclei') for item in ls_rim]
    ls_nuc_ring = [item.replace('Rim','Ring') for item in ls_nuc_rim]
    ls_nuc_ring.append('scene')
    ls_nuc_rim.append('scene')
    df_scene = add_scene(df_mi)
    ls_nuc_ring = sorted(set(df_scene.columns).intersection(set(ls_nuc_ring)))
    #quntiles
    df_bg =  df_scene.loc[:,ls_nuc_ring].groupby('scene').quantile(0) 
    df_bg.columns = [f'{item}' for item in df_bg.columns]
    for q in np.arange(0,1,.1):
        df_quantile = df_scene.loc[:,ls_nuc_ring].groupby('scene').quantile(q)
        df_bg = df_bg.merge(df_quantile,left_index=True, right_index=True, suffixes=('',f'_{str(int(q*10))}'))
        print(q)
        print(f'_{str(int(q*10))}')
    #mean
    df_quantile = df_scene.loc[:,ls_nuc_ring].groupby('scene').mean()
    df_bg = df_bg.merge(df_quantile,left_index=True, right_index=True, suffixes=('','_mean'))
    #drop duplicate
    ls_nuc_ring.remove('scene')
    df_bg = df_bg.loc[:,~df_bg.columns.isin(ls_nuc_ring)]
    return(df_bg)

def exposure_norm(df_mi,df_t,d_factor={'c1':10,'c2':30,'c3':200,'c4':500,'c5':500}):
    """
    normalizes to standard exposure times
    input: mean intensity, and metadata table with exposure time
    """
    df_norm = pd.DataFrame()
    ls_columns =  [item.split('_')[0] for item in df_mi.columns.tolist()]
    ls_column_mi = df_mi.columns.tolist()
    for idx, s_column in enumerate(ls_columns):

        s_marker = s_column.split('_')[0]
        i_exp = df_t.loc[s_column,'exposure']
        print(f'Processing exposure time for {s_column}: {i_exp}')
        print(f'Processing mean intensity {ls_column_mi[idx]}')
        i_factor = d_factor[df_t.loc[s_column,'colors']]
        se_exp = df_mi.loc[:,ls_column_mi[idx]]
        df_norm[ls_column_mi[idx]] = se_exp/i_exp*i_factor
    return(df_norm)

def af_subtract(df_norm,df_t,d_channel={'c2':['L488','L488'],'c3':['L555','L555'],'c4':['L647','L647'],'c5':['L750','L750']},ls_exclude=[]):
    """
    given an exposure normalized dataframe, metadata with biomarker location, and a dictionary of background channels, subtracts
    correct background intensity from each cell
    input:
    d_channel = dictionary, key is color i.e. 'c2', value is list of ['Ring','Nuclei']
    ls_exclude = markers to not subtract
    output:
    df_mi_sub,ls_sub,ls_record
    """
    #generate list of background markers needed for subtraction
    lls_d_channel = []
    for s_key,ls_item in d_channel.items():
        lls_d_channel = lls_d_channel + [ls_item]
    ls_background = []
    for ls_channel in lls_d_channel:
        ls_background = ls_background + [f'{ls_channel[0]}_Ring']
        ls_background = ls_background + [f'{ls_channel[1]}_Nuclei']
    se_background = set(ls_background)
    se_exclude = set([item + '_Ring' for item in ls_exclude] + [item + '_Nuclei' for item in ls_exclude]).intersection(set(df_norm.columns.tolist()))
    se_all = set(df_norm.columns.tolist())
    se_sub = se_all - se_background - se_exclude
    ls_sub = list(se_sub)

    #subtract AF channels
    df_mi_sub = pd.DataFrame()
    
    ls_record = []
    for s_marker_loc in ls_sub:
        print(s_marker_loc)
        s_marker = s_marker_loc.split('_')[0]
        s_loc = s_marker_loc.split('_')[1]
        s_channel = df_t.loc[s_marker,'colors']
        if s_channel == 'c1':
            df_mi_sub[s_marker_loc] = df_norm.loc[:,s_marker_loc]
            continue
        if s_loc =='Nuclei':
            s_AF = d_channel[s_channel][1]
        elif s_loc == 'Ring':
            s_AF = d_channel[s_channel][0]
        else:
            print('Error: location must be Ring or Nucleus')
        s_AF_loc = s_AF + '_' + s_loc
        df_mi_sub[s_marker_loc] = df_norm.loc[:,s_marker_loc] - df_norm.loc[:,s_AF_loc]
        print(f'From {s_marker_loc} subtracting {s_AF_loc}')
        ls_record = ls_record + [f'From {s_marker_loc} subtracting {s_AF_loc}\n']
    for s_marker in sorted(se_exclude):
        ls_record = ls_record + [f'From {s_marker} subtracting None\n']
    df_mi_sub[sorted(se_exclude)] = df_norm.loc[:,sorted(se_exclude)]
    #f = open(f"AFsubtractionData.txt", "w")
    #f.writelines(ls_record)
    #f.close()
    #error check
    print('AF subtraction not performed for the following markers:')
    print(set(df_t.index) - set(ls_sub))
    
    return(df_mi_sub,ls_sub,ls_record)

def plot_subtraction(df_norm,df_sub,ls_scene=None):
    """
    makes scatterplots of each marker, subtracted versus original meanintensity per cell, to judge subtraction effectiveness
    """
    if ls_scene == None:
        ls_scene = list(set(df_norm.scene))
    ls_marker = df_sub.columns.tolist()
    ls_marker.remove('scene')
    ls_scene.sort()
    for s_marker in ls_marker:
        print(f'Plotting {s_marker}')
        fig, ax = plt.subplots(2,(len(ls_scene)+1)//2, figsize = (12,4))
        ax = ax.ravel()
        ax_num = -1
        for s_scene in ls_scene:
            df_subtracted = df_sub[df_sub.scene==s_scene]
            df_original = df_norm[df_norm.scene==s_scene]
            ax_num = ax_num + 1
            ax[ax_num].scatter(x=df_original.loc[:,s_marker],y=df_subtracted.loc[:,s_marker],s=1,alpha=0.8)
            ax[ax_num].set_title(s_scene,{'fontsize': 10,'verticalalignment': 'center'})
            fig.text(0.5, 0.01, s_marker, ha='center') 
            fig.text(0.6, 0.01, 'Original', ha='center') 
            fig.text(0.01, 0.6, 'Subtracted', va='center', rotation='vertical')
            plt.tight_layout()
            fig.savefig(f'{s_marker}_NegativevsOriginal.png')

def output_subtract(df_sub,df_t,d_factor={'c1':10,'c2':30,'c3':200,'c4':500,'c5':500}):
    """
    this un-normalizes by exposure time to output a new dataframe of AF subtracted cells for analysis
    """
    ls_sub = df_sub.columns.tolist()
    result = any(elem == 'scene' for elem in ls_sub)
    if result:
        ls_sub.remove('scene')
        df_sub = df_sub.drop(columns='scene')
    else:
        print('no scene column')
    df_mi_zero = df_sub.clip(lower = 0)
    df_mi_factor = pd.DataFrame()
    for s_sub in ls_sub:
        s_dft_index = s_sub.split('_')[0]
        i_reverse_factor = df_t.loc[s_dft_index,'exposure']/d_factor[df_t.loc[s_dft_index,'colors']]
        df_mi_factor[s_sub] = df_mi_zero.loc[:,s_sub]*i_reverse_factor
    return df_mi_factor

def af_subtract_images(df_t,d_channel={'c2':['L488','L488'],'c3':['L555','L555'],'c4':['L647','L647'],'c5':['L750','L750']},s_dapi='DAPI11_Nuclei',b_mkdir=True):
    """
    This code loads 16 bit grayscale tiffs, performs AF subtraction of channels/rounds defined by the user, and outputs 8 bit AF subtracted tiffs for visualization.
    The data required is:
    1. The RoundsCyclesTable.txt with the location (Nucleus/Ring) specified (not All), and real expsure times
    2. 16 bit grayscale tiff images following Koei's naming convention (script processes the list of folders ls_folder) 
    Note: name of folder can be anything
    """
    #generate list of markers needing subtraction
    lls_d_channel = []
    for s_key in d_channel:
        lls_d_channel = lls_d_channel + [d_channel[s_key]]
    ls_background = []
    for ls_channel in lls_d_channel:
        ls_background = ls_background + [f'{ls_channel[0]}_Ring']
        ls_background = ls_background + [f'{ls_channel[1]}_Nuclei']
    se_background = set(ls_background)
    se_all = set(df_t.index)
    se_sub = se_all - se_background
    ls_sub = list(se_sub)
    #ls_sub.remove(s_dapi) #don't need line if s_DAPI is c1
    #subtract images
    #os.makedirs('8bit/', exist_ok=True)
    if b_mkdir:
        os.mkdir('8bit')
    ls_image = os.listdir()
    ls_slide = []
    ls_image_org = []
    for s_image in ls_image:
        if s_image.find('_ORG.tif')>-1:
            #make a list of slides/scenes in the folder
            s_slide = s_image.split('_')[2]
            ls_slide = ls_slide + [s_slide]
            #make a list of all original images in the folder
            ls_image_org = ls_image_org + [s_image]
    ls_slide = list(set(ls_slide))
    #process each slide in the folder
    for s_slide in ls_slide:
        print(f'Processing {s_slide}')
        df_t['image'] = 'NA'
        ls_dapi = []
        
        for s_image in ls_image_org:
            
            #grab all original images with slide/scene name
            if s_image.find(s_slide) > -1:
        
                #add matching image name to df_t (fore specific slide/scene, dapi not included)
                s_round = s_image.split('Registered-')[1].split('_')[0]
                s_color = s_image.split('Scene-')[1].split('_')[1]
                s_index = df_t[(df_t.rounds==s_round) & (df_t.color==s_color)].index
                df_t.loc[s_index,'image'] = s_image
                if s_color == 'c1':
                    ls_dapi = ls_dapi + [s_image]
        #subtract images
        ls_record = []
        for s_marker_loc in ls_sub:
            s_marker = s_marker_loc.split('_')[0]
            s_loc = s_marker_loc.split('_')[1]
            s_rounds= df_t.loc[s_marker_loc,'rounds']
            s_channel = df_t.loc[s_marker_loc,'color']
            if s_channel == 'c1':
                print(f'{s_marker_loc} is DAPI')
                continue
            elif s_loc =='Nuclei':
                s_AF = d_channel[s_channel][1]
            elif s_loc == 'Ring':
                s_AF = d_channel[s_channel][0]
            else:
                print('Error: location must be Ring or Nucleus')
            s_AF_loc = s_AF + '_' + s_loc 
            print(f'From {s_marker_loc} subtracting {s_AF_loc}')
            s_image = df_t.loc[s_marker_loc,'image']
            s_background = df_t.loc[s_AF_loc,'image']
            a_img = skimage.io.imread(s_image)
            a_AF = skimage.io.imread(s_background)
            #divide each image by exposure time
            #subtract 1 ms AF from 1 ms signal
            #multiply by original image exposure time
            a_sub = (a_img/df_t.loc[s_marker_loc,'exposure'] - a_AF/df_t.loc[s_AF_loc,'exposure'])*df_t.loc[s_marker_loc,'exposure']

            ls_record = ls_record + [f'From {s_marker_loc} subtracting {s_AF_loc}\n']
            #make all negative numbers into zero
            a_zero = a_sub.clip(min=0,max=a_sub.max())
            a_zero_8bit = (a_zero/256).astype(np.uint8)
            s_fname = f"8bit/{s_rounds}_{s_marker}_{s_slide}_{s_channel}_8bit.tif"
            skimage.io.imsave(s_fname,a_zero_8bit)
        f = open(f"8bit/AFsubtractionImages.txt", "w")
        f.writelines(ls_record)
        f.close()
        #save 8 bit dapis
        for s_dapi in ls_dapi:
            a_img = skimage.io.imread(s_dapi)
            a_zero_8bit = (a_img/256).astype(np.uint8)
            s_marker = 'DAPI'
            s_channel = 'c1'
            s_round =  s_dapi.split('Registered-')[1].split('_')[0]
            s_fname = f"8bit/{s_round}_{s_marker}_{s_slide}_{s_channel}_8bit.tif"
            skimage.io.imsave(s_fname,a_zero_8bit)

def round_overlays():
    """
    output multipage tiffs with five channels per round
    """
    os.chdir('./8bit')
    ls_image = os.listdir()
    ls_slide = []
    ls_image_org = []
    ls_round = []

    for s_image in ls_image:
        if s_image.find('8bit.tif') > -1:
            #make a list of slides/scenes
            #also make list of rounds
            s_slide = s_image.split('_')[2]
            ls_slide = ls_slide + [s_slide]
            ls_image_org = ls_image_org + [s_image]
            s_round = s_image.split('_')[0]
            ls_round = ls_round + [s_round]
        ls_slide = list(set(ls_slide))
        ls_round = list(set(ls_round))
    for s_slide in ls_slide:
        print(f'Processing {s_slide}')
        for s_round in ls_round:
            d_overlay = {}
            ls_color_round = []
            for s_image in ls_image_org:
                if s_image.find(s_slide) > -1:
                    if s_image.find(f'{s_round}_') == 0:
                        s_color = s_image.split('_')[3]
                        d_overlay.update({s_color:s_image})
                        s_image_round = s_image
            a_size = skimage.io.imread(s_image_round)
            a_overlay = np.zeros((len(d_overlay),a_size.shape[0],a_size.shape[1]),dtype=np.uint8)
            s_biomarker_all = ''
            i = -1 
            for s_color in sorted(d_overlay.keys()):
                i = i + 1
                s_overlay= d_overlay[s_color]
                s_biomarker = s_overlay.split('_')[1] + '.'
                s_biomarker_all = s_biomarker_all + s_biomarker
                a_channel = skimage.io.imread(s_overlay)
                a_overlay[i,:,:] = a_channel
            s_biomarker_all = s_biomarker_all[:-1]
            #this works. Open in image j. use Image/Color/Make Composite. Then use 
            #Image/Color/Channels Tool to turn on and off channels
            #use Image/Adjust/Brightness/Contrast to adjust 
            with skimage.external.tifffile.TiffWriter(f'{s_round}_{s_biomarker_all}_{s_slide}_overlay.tiff', imagej=True) as tif:
                for i in range(a_overlay.shape[0]):
                    tif.save(a_overlay[i])
    os.chdir('..')

def custom_overlays(d_combos, df_img, df_dapi):
    """
    output custon multi page tiffs according to dictionary, with s_dapi as channel 1 in each overlay
    BUG with 53BP1
    d_combos = {'Immune':{'CD45', 'PD1', 'CD8', 'CD4', 'CD68', 'FoxP3','GRNZB','CD20','CD3'},
    'Stromal':{'Vim', 'aSMA', 'PDPN', 'CD31', 'ColIV','ColI'},
    'Differentiation':{'CK19', 'CK7','CK5', 'CK14', 'CK17','CK8'},
    'Tumor':{'HER2', 'Ecad', 'ER', 'PgR','Ki67','PCNA'},
    'Proliferation':{'EGFR','CD44','AR','pHH3','pRB'}, 
    'Functional':{'pS6RP','H3K27','H3K4','cPARP','gH2AX','pAKT','pERK'},
    'Lamins':{'LamB1','LamAC', 'LamB2'}}
    """
    #os.chdir('./AFSubtracted')

    ls_slide = list(set(df_img.scene))
    #now make overlays
    for s_slide in ls_slide:
        print(f'Processing {s_slide}')
        df_slide = df_img[df_img.scene==s_slide]
        s_image_round = (df_dapi[df_dapi.scene == s_slide]).index[0]
        if len((df_dapi[df_dapi.scene == s_slide]).index) == 0:
            print('Error: dapi not found')
        elif len((df_dapi[df_dapi.scene == s_slide]).index) > 1:
            print('Error: too many dapi images found')
        else:
            print(s_image_round)
        #exclude any missing biomarkers
        es_all = set(df_slide.marker)
        if len(list(set(df_img.imagetype)))==1:
            s_imagetype = list(set(df_img.imagetype))[0]
            print(s_imagetype)
        else:
            print('Error: more than one image type)')
        for s_type in d_combos:
            d_overlay = {}
            es_combos = d_combos[s_type]
            es_combos_shared = es_combos.intersection(es_all)
            for idx, s_combo in enumerate(sorted(es_combos_shared)):
                s_filename = (df_slide[df_slide.marker==s_combo]).index[0]
                if len((df_slide[df_slide.marker==s_combo]).index) == 0:
                    print('Error: marker not found')
                elif len((df_slide[df_slide.marker==s_combo]).index) > 1:
                    print('Error: too many marker images found')
                else:
                    print(s_filename)
                d_overlay.update({s_combo:s_filename})
            d_overlay.update({'1AAADAPI':s_image_round})
            a_size = skimage.io.imread(s_image_round)
            a_overlay = np.zeros((len(d_overlay),a_size.shape[0],a_size.shape[1]),dtype=np.uint8)
            s_biomarker_all = ''
            i = -1 
            for s_color in sorted(d_overlay.keys()):
                i = i + 1
                s_overlay= d_overlay[s_color]
                s_biomarker = s_color.split('1AAA')[0] + '.'
                s_biomarker_all = s_biomarker_all + s_biomarker
                a_channel = skimage.io.imread(s_overlay)
                if s_imagetype=='ORG':
                    a_channel = (a_channel/256).astype(np.uint8)
                    print('covert to 8 bit')
                a_overlay[i,:,:] = a_channel
            s_biomarker_all = s_biomarker_all[1:-1]
            #this works. Open in image j. use Image/Color/Make Composite. Then use 
            #Image/Color/Channels Tool to turn on and off channels
            #use Image/Adjust/Brightness/Contrast to adjust 
            with skimage.external.tifffile.TiffWriter(f'./{s_type}_{((df_dapi[df_dapi.scene==s_slide]).marker[0])}.{s_biomarker_all}_{s_slide}_overlay.tiff', imagej=True) as tif:
                for i in range(a_overlay.shape[0]):
                    tif.save(a_overlay[i])
            print(f'saved {s_type}')

def custom_crop_overlays(d_combos,d_crop, df_img,s_dapi, tu_dim=(1000,1000)): #df_dapi,
    """
    output custon multi page tiffs according to dictionary, with s_dapi as channel 1 in each overlay
    BUG with 53BP1
    d_crop : {slide_scene : (x,y) coord
    tu_dim = (width, height)
    d_combos = {'Immune':{'CD45', 'PD1', 'CD8', 'CD4', 'CD68', 'FoxP3','GRNZB','CD20','CD3'},
    'Stromal':{'Vim', 'aSMA', 'PDPN', 'CD31', 'ColIV','ColI'},
    'Differentiation':{'CK19', 'CK7','CK5', 'CK14', 'CK17','CK8'},
    'Tumor':{'HER2', 'Ecad', 'ER', 'PgR','Ki67','PCNA'},
    'Proliferation':{'EGFR','CD44','AR','pHH3','pRB'}, 
    'Functional':{'pS6RP','H3K27','H3K4','cPARP','gH2AX','pAKT','pERK'},
    'Lamins':{'LamB1','LamAC', 'LamB2'}}
    """
    #os.chdir('./AFSubtracted')

    ls_slide = list(set(df_img.scene))
    #now make overlays
    for s_slide, xy_cropcoor in d_crop.items():
        print(f'Processing {s_slide}')
        df_slide = df_img[df_img.scene==s_slide]
        s_image_round = df_slide[df_slide.marker==s_dapi.split('_')[0]].index[0]
        if len(df_slide[df_slide.marker==s_dapi.split('_')[0]].index) == 0:
            print('Error: dapi not found')
        elif len(df_slide[df_slide.marker==s_dapi.split('_')[0]].index) > 1:
            print('Error: too many dapi images found')
        else:
            print(s_image_round)
        #exclude any missing biomarkers
        es_all = set(df_slide.marker)
        if len(list(set(df_img.imagetype)))==1:
            s_imagetype = list(set(df_img.imagetype))[0]
            print(s_imagetype)
        else:
            print('Error: more than one image type)')
        for s_type, es_combos in d_combos.items():
            d_overlay = {}
            es_combos_shared = es_combos.intersection(es_all)
            for idx, s_combo in enumerate(sorted(es_combos_shared)):
                s_filename = (df_slide[df_slide.marker==s_combo]).index[0]
                if len((df_slide[df_slide.marker==s_combo]).index) == 0:
                    print('Error: marker not found')
                elif len((df_slide[df_slide.marker==s_combo]).index) > 1:
                    print('Error: too many marker images found')
                else:
                    print(s_filename)
                d_overlay.update({s_combo:s_filename})
            d_overlay.update({'1AAADAPI':s_image_round})
            a_size = skimage.io.imread(s_image_round)
            #crop 
            a_crop = a_size[(xy_cropcoor[1]):(xy_cropcoor[1]+tu_dim[1]),(xy_cropcoor[0]):(xy_cropcoor[0]+tu_dim[0])]
            a_overlay = np.zeros((len(d_overlay),a_crop.shape[0],a_crop.shape[1]),dtype=np.uint8)
            s_biomarker_all = ''
            i = -1 
            for s_color in sorted(d_overlay.keys()):
                i = i + 1
                s_overlay= d_overlay[s_color]
                s_biomarker = s_color.split('1AAA')[0] + '.'
                s_biomarker_all = s_biomarker_all + s_biomarker
                a_size = skimage.io.imread(s_overlay)
                #crop 
                a_channel = a_size[(xy_cropcoor[1]):(xy_cropcoor[1]+tu_dim[1]),(xy_cropcoor[0]):(xy_cropcoor[0]+tu_dim[0])]
                if s_imagetype=='ORG':
                    a_channel = (a_channel/256).astype(np.uint8)
                    print('covert to 8 bit')
                a_overlay[i,:,:] = a_channel
            s_biomarker_all = s_biomarker_all[1:-1]
            #this works. Open in image j. use Image/Color/Make Composite. Then use 
            #Image/Color/Channels Tool to turn on and off channels
            #use Image/Adjust/Brightness/Contrast to adjust 
            with skimage.external.tifffile.TiffWriter(f'./{s_type}_{s_dapi.split("_")[0]}.{s_biomarker_all}_{s_slide}_x{xy_cropcoor[0]}y{xy_cropcoor[1]}_overlay.tiff', imagej=True) as tif:
                for i in range(a_overlay.shape[0]):
                    tif.save(a_overlay[i])
            print(f'saved {s_type}')

def make_thresh_df(df_out,ls_drop=None):
    """
    makes a thresholding csv matching the output dataframe (df_out)'s scenes and biomarkers
    """
    ls_scene = list(set(df_out.scene))
    ls_scene.append('global_manual')
    ls_scene.sort()
    ls_biomarker = df_out.columns.tolist()
    ls_biomarker.remove('scene')
    if ls_drop != None:
        for s_drop in ls_drop:
            ls_biomarker.remove(s_drop)
    ls_manual = []
    for s_biomarker in ls_biomarker:
        s_marker = s_biomarker.split('_')[0] + '_manual'
        ls_manual.append(s_marker)
    ls_manual.sort()
    df_thresh = pd.DataFrame(index=ls_scene,columns=ls_manual)
    #df_thresh_t = df_thresh.transpose()
    return(df_thresh)

def check_seg(s_sample= 'sampleID',ls_find=['Cell Segmentation Full Color'], i_rows=2, t_figsize=(20,10)):
    """
    This script makes overviews of all the specified segmentation images of guillaumes ouput images
    in a big folder (slides prepared for segmentation for example)
    Input: ls_find = list of images to view
     i_rows = number or rows in figure
     t_figsize = (x, y) in inches size of figure
     b_mkdir = boolean whether to make a new Check_Registration folder (deprecated)
    Output: dictionary with {slide_color:number of rounds found}
     images of all rounds of a certain slide_color
    """
    d_result = {}
    #if b_mkdir:
    #        os.mkdir(f'./Check_Registration')
    for s_find in ls_find:
        #find all dapi slides
        ls_dapis = []
        for s_dir in os.listdir():
            if s_dir.find(s_find) > -1:
                ls_dapis = ls_dapis + [s_dir]
        ls_dapis.sort()
        
        #find all unique scenes
        ls_scene_long = []
        for s_dapi in ls_dapis:
            ls_scene_long = ls_scene_long + [(s_dapi.split('-')[0])]
        ls_scene = list(set(ls_scene_long))
        ls_scene.sort()
        fig,ax = plt.subplots(i_rows,(len(ls_scene)+(i_rows-1))//i_rows, figsize = t_figsize, squeeze=False)
        ax = ax.ravel()
        for idx, s_scene in enumerate(ls_scene):
            print(f'Processing {s_scene}')
            im_low = skimage.io.imread(ls_dapis[idx])#,plugin='simpleitk'
            im = skimage.exposure.rescale_intensity(im_low,in_range=(np.quantile(im_low,0.02),np.quantile(im_low,0.98)+np.quantile(im_low,0.98)/2))
            im = skimage.transform.rescale(im, 0.25, anti_aliasing=False)
            ax[idx].imshow(im) #, cmap='gray'
            ax[idx].set_title(s_scene,{'fontsize':12})
        plt.tight_layout()
        #fig.savefig(f'../Check_Registration/{s_sample}_{s_find}.png')
        d_result.update({f'{s_sample}_{s_find}.png':fig})
    return(d_result)
