####
# title: analyze.py
#
# language: Python3.6
# date: 2019-05-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 library to visualize cyclic data and analysis
####

#load libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io, segmentation
import tifffile
import copy
import napari
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

#napari
def load_crops(viewer,s_crop,s_tissue):
    ls_color = ['blue','green','yellow','red','cyan','magenta','gray','green','yellow','red','cyan','magenta',
     'gray','gray','gray','gray','gray','gray','gray','gray']
    print(s_crop)
    #viewer = napari.Viewer()
    for s_file in os.listdir():
        if s_file.find(s_tissue)>-1:
            if s_file.find(s_crop) > -1:
                if s_file.find('ome.tif') > -1:
                    with tifffile.TiffFile(s_file) as tif:
                        array = tif.asarray()
                        omexml_string = tif.ome_metadata
                        for idx in range(array.shape[0]):
                            img = array[idx]
                            i_begin = omexml_string.find(f'Channel ID="Channel:0:{idx}" Name="')
                            i_end = omexml_string[i_begin:].find('" SamplesPerPixel')
                            s_marker = omexml_string[i_begin + 31:i_begin + i_end]
                            viewer.add_image(img,name=s_marker,rgb=False,visible=False,blending='additive',colormap=ls_color[idx],contrast_limits = (np.quantile(img,0),(np.quantile(img,0.9999)+1)*1.5))
                elif s_file.find('SegmentationBasins') > -1:
                    label_image = io.imread(s_file)
                    viewer.add_labels(label_image, name='cell_seg',blending='additive',visible=False)
                    cell_boundaries = segmentation.find_boundaries(label_image,mode='outer')
                    viewer.add_labels(cell_boundaries,blending='additive')
                else:
                    label_image = np.array([])
                    print('')
    return(label_image)

def pos_label(viewer,df_pos,label_image,s_cell):
    '''
    df_pos = boolean dataframe, s_cell = marker name 
    '''
    #s_cell = df_pos.columns[df_pos.columns.str.contains(f'{s_cell}_')][0]
    #get rid of extra cells (filtered by DAPI, etc)
    li_index = [int(item.split('_')[-1].split('cell')[1]) for item in df_pos.index]
    label_image_cell = copy.deepcopy(label_image)
    label_image_cell[~np.isin(label_image_cell, li_index)] = 0
    li_index_cell = [int(item.split('_')[-1].split('cell')[1]) for item in df_pos[df_pos.loc[:,s_cell]==True].index]
    label_image_cell[~np.isin(label_image_cell,li_index_cell )] = 0
    viewer.add_labels(label_image_cell, name=f'{s_cell.split("_")[0]}_seg',blending='additive',visible=False)
    return(label_image_cell)

#jupyter notbook
#load manual thresholds
def new_thresh_csv(df_mi,d_combos):
    #make thresh csv's
    df_man = pd.DataFrame(index= ['global']+ sorted(set(df_mi.slide_scene)))
    for s_type, es_marker in d_combos.items():
        for s_marker in sorted(es_marker):
            df_man[s_marker] = ''
    return(df_man)

def load_thresh_csv(s_sample):
    #load
    df_man = pd.read_csv(f'thresh_JE_{s_sample}.csv',header=0,index_col = 0)
    #reformat the thresholds data and covert to 16 bit 
    ls_index = df_man.index.tolist()
    ls_index.remove('global')
    df_thresh = pd.DataFrame(index = ls_index)
    ls_marker = df_man.columns.tolist()
    for s_marker in ls_marker:
        df_thresh[f'{s_marker}_global'] = df_man[df_man.index=='global'].loc['global',f'{s_marker}']*256
        df_thresh[f'{s_marker}_local'] = df_man[df_man.index!='global'].loc[:,f'{s_marker}']*256

    df_thresh.replace(to_replace=0, value = 12, inplace=True)
    return(df_thresh)

def threshold_postive(df_thresh,df_mi):
    '''
    #make positive dataframe to check threhsolds #start with local, and if its not there, inesrt the global threshold
    #note, this will break if there are two biomarker locations #
    '''
    ls_scene = sorted(df_thresh.index.tolist())
    ls_sub = df_mi.columns[df_mi.dtypes=='float64'].tolist()
    ls_other = []
    df_pos= pd.DataFrame()
    d_thresh_record= {}
    for s_scene in ls_scene:
        ls_index = df_mi[df_mi.slide_scene==s_scene].index
        df_scene = pd.DataFrame(index=ls_index)
        for s_marker_loc in ls_sub:
            s_marker = s_marker_loc.split('_')[0]
            # only threshold markers in .csv
            if len(set([item.split('_')[0] for item in df_thresh.columns]).intersection({s_marker})) != 0:
                #first check if local threshold exists
                if df_thresh[df_thresh.index==s_scene].isna().loc[s_scene,f'{s_marker}_local']==False:
                    #local
                    i_thresh = df_thresh.loc[s_scene,f'{s_marker}_local']
                    df_scene.loc[ls_index,s_marker_loc] = df_mi.loc[ls_index,s_marker_loc] >= i_thresh
                #otherwise use global
                elif df_thresh[df_thresh.index==s_scene].isna().loc[s_scene,f'{s_marker}_global']==False:
                    i_thresh = df_thresh.loc[s_scene,f'{s_marker}_global']
                    df_scene.loc[ls_index,s_marker_loc] = df_mi.loc[ls_index,s_marker_loc] >= i_thresh
                else:
                    ls_other = ls_other + [s_marker]
                    i_thresh = np.NaN
                d_thresh_record.update({f'{s_scene}_{s_marker}':i_thresh})
            else:
                ls_other = ls_other + [s_marker]
        df_pos = df_pos.append(df_scene)
    print(f'Did not threshold {set(ls_other)}')
    return(d_thresh_record,df_pos)

def plot_positive(s_type,d_combos,df_pos,d_thresh_record,df_xy,b_save=True):
    ls_color = sorted(d_combos[s_type])
    ls_bool = [len(set([item.split('_')[0]]).intersection(set(ls_color)))==1 for item in df_pos.columns]
    ls_color = df_pos.columns[ls_bool].tolist()
    ls_scene = sorted(set(df_xy.slide_scene))
    ls_fig = []
    for s_scene in ls_scene:
        #negative cells = all cells even before dapi filtering
        df_neg = df_xy[(df_xy.slide_scene==s_scene)]
        #plot
        fig, ax = plt.subplots(2, ((len(ls_color))+1)//2, figsize=(18,12)) #figsize=(18,12)
        ax = ax.ravel()
        for ax_num, s_color in enumerate(ls_color):
            s_marker = s_color.split('_')[0]
            s_min = d_thresh_record[f"{s_scene}_{s_marker}"]
            #positive cells = positive cells based on threshold
            ls_pos_index = (df_pos[df_pos.loc[:,s_color]]).index
            df_color_pos = df_neg[df_neg.index.isin(ls_pos_index)]
            if len(df_color_pos)>=1:
                #plot negative cells
                ax[ax_num].scatter(data=df_neg,x='DAPI_X',y='DAPI_Y',color='silver',s=1)
                #plot positive cells
                ax[ax_num].scatter(data=df_color_pos, x='DAPI_X',y='DAPI_Y',color='DarkBlue',s=.5)
                      
                ax[ax_num].axis('equal')
                ax[ax_num].set_ylim(ax[ax_num].get_ylim()[::-1])
                ax[ax_num].set_title(f'{s_marker} min={int(s_min)} ({len(df_color_pos)} cells)')
            else:
                ax[ax_num].set_title(f'{s_marker} min={(s_min)} ({(0)} cells')
        fig.suptitle(s_scene)
        ls_fig.append(fig)
        if b_save:
            fig.savefig(f'./SpatialPlots/{s_scene}_{s_type}_manual.png')
    return(ls_fig)

#gating analysis
def prop_positive(df_data,s_cell,s_grouper):
    #df_data['countme'] = True
    df_cell = df_data.loc[:,[s_cell,s_grouper,'countme']].dropna()
    df_prop = (df_cell.groupby([s_cell,s_grouper]).countme.count()/df_cell.groupby([s_grouper]).countme.count()).unstack().T
    return(df_prop)

def prop_clustermap(df_prop,df_annot,i_thresh,lut,figsize=(10,5)):
    for s_index in df_prop.index:
        s_subtype = df_annot.loc[s_index,'ID'] #
        df_prop.loc[s_index, 'ID'] = s_subtype
    species = df_prop.pop("ID")
    row_colors = species.map(lut)

    #clustermap plot wihtout the low values -drop less than i_threh % of total
    df_plot = df_prop.fillna(0)
    if i_thresh > 0:
        df_plot_less = df_plot.loc[:,df_plot.sum()/len(df_plot) > i_thresh]
    i_len = len(df_prop)
    i_width = len(df_plot_less.columns)
    g = sns.clustermap(df_plot_less,figsize=figsize,cmap='viridis',row_colors=row_colors)
    return(g,df_plot_less)

def prop_barplot(df_plot_less,s_cell,colormap="Spectral",figsize=(10,5),b_sort=True):
    i_len = len(df_plot_less)
    i_width = len(df_plot_less.columns)
    fig,ax = plt.subplots(figsize=figsize)
    if b_sort:
        df_plot_less = df_plot_less.sort_index(ascending=False)
    df_plot_less.plot(kind='barh',stacked=True,width=.9, ax=ax,colormap=colormap)
    ax.set_title(s_cell)
    ax.set_xlabel('Fraction Positive')
    ax.legend(bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    return(fig)

def plot_color_leg(lut,figsize = (2.3,3)):
    #colors
    series = pd.Series(lut)
    df_color = pd.DataFrame(index=range(len(series)),columns=['subtype','color'])

    series.sort_values()
    df_color['subtype'] = series.index
    df_color['value'] = 1
    df_color['color'] = series.values

    fig,ax = plt.subplots(figsize = figsize,dpi=100)
    df_color.plot(kind='barh',x='subtype',y='value',width=1,legend=False,color=df_color.color,ax=ax)
    ax.set_xticks([])
    ax.set_ylabel('')
    ax.set_title(f'subtype')
    plt.tight_layout()
    return(fig)

#cluster analysis

def cluster_kmeans(df_mi,ls_columns,k,b_sil=False):
    '''
    log2 transform, zscore and kmens cluster
    '''
    df_cluster_norm = df_mi.loc[:,ls_columns]
    df_cluster_norm_one = df_cluster_norm + 1
    df_cluster = np.log2(df_cluster_norm_one)

    #select figure size
    i_len = k
    i_width = len(df_cluster.columns)

    #scale date
    df_scale = scale(df_cluster)

    #kmeans cluster
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df_scale)
    df_cluster.columns = [item.split('_')[0] for item in df_cluster.columns]
    df_cluster[f'K{k}'] = list(kmeans.labels_)
    g = sns.clustermap(df_cluster.groupby(f'K{k}').mean(),cmap="RdYlGn_r",z_score=1,figsize=(3+i_width/3,3+i_len/3))
    if b_sil:
        score = silhouette_score(X = df_scale, labels=list(kmeans.labels_))
    else:
        score = np.nan
    return(g,df_cluster,score)

def plot_clusters(df_cluster,df_xy,s_num='many'):
    s_type = df_cluster.columns[df_cluster.dtypes=='int64'][0]
    print(s_type)
    ls_scene = sorted(set(df_cluster.slide_scene))
    ls_color = sorted(set(df_cluster.loc[:,s_type].dropna()))
    d_fig = {}
    for s_scene in ls_scene:
        #negative cells = all cells even before dapi filtering
        df_neg = df_xy[(df_xy.slide_scene==s_scene)]
        #plot
        if s_num == 'many':
            fig, ax = plt.subplots(3, ((len(ls_color))+2)//3, figsize=(18,12),dpi=200)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(7,4),dpi=200)	
        ax = ax.ravel()
        for ax_num, s_color in enumerate(ls_color):
            s_marker = s_color
            #positive cells = poitive cells based on threshold
            ls_pos_index = (df_cluster[df_cluster.loc[:,s_type]==s_color]).index
            df_color_pos = df_neg[df_neg.index.isin(ls_pos_index)]
            if len(df_color_pos)>=1:
                #plot negative cells
                ax[ax_num].scatter(data=df_neg,x='DAPI_X',y='DAPI_Y',color='silver',s=1)
                #plot positive cells
                ax[ax_num].scatter(data=df_color_pos, x='DAPI_X',y='DAPI_Y',color='DarkBlue',s=.5)
                  
                ax[ax_num].axis('equal')
                ax[ax_num].set_ylim(ax[ax_num].get_ylim()[::-1])
                if s_num == 'many':
                    ax[ax_num].set_xticklabels('')
                    ax[ax_num].set_yticklabels('')
                ax[ax_num].set_title(f'{s_color} ({len(df_color_pos)} cells)')
            else:
                ax[ax_num].set_xticklabels('')
                ax[ax_num].set_yticklabels('')
                ax[ax_num].set_title(f'{s_color}  ({(0)} cells')
        
        fig.suptitle(s_scene)
        d_fig.update({s_scene:fig})
    return(d_fig)
