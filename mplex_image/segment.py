####
# title: segment.py
#
# language: Python3.7
# date: 2020-06-00
# license: GPL>=v3
# author: Jenny
#
# description:
#   python3 script for cell segmentation
####
import time
import cellpose
from cellpose import models
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

import os
import skimage
import pandas as pd
import numpy as np
import sys 
import scipy
from scipy import stats
from scipy import ndimage as ndi
from skimage import io, filters
from skimage import measure, segmentation, morphology
from numba import jit, types
from numba.extending import overload
from numba.experimental import jitclass
import numba
import mxnet as mx 
import stat
from mxnet import nd
from mplex_image import preprocess

#set src path (CHANGE ME)
s_src_path = '/home/groups/graylab_share/OMERO.rdsStore/engje/Data/cmIF'

#functions

def gpu_device():
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu())
        mx_gpu = mx.gpu()
    except mx.MXNetError:
        return None
    return mx_gpu

def cellpose_nuc(key,dapi,diameter=30):
    '''
    smallest nuclei are about 9 pixels, lymphocyte is 15 pixels, tumor is 25 pixels
    using 20 can capture large tumor cells, without sacrificing smaller cells,
    '''
    try:
        nd_array = mx.nd.array([1, 2, 3], ctx=mx.gpu())
        print(nd_array)
        mx_gpu = mx.gpu()
    except mx.MXNetError:
        print('Mxnet error')
        mx_gpu = None
    model = models.Cellpose(model_type='nuclei',device=mx_gpu)
    newkey = f"{key.split(' - Z')[0]} nuclei{diameter}"
    print(f"modelling {newkey}")
    channels = [0,0] 
    print(f'Minimum nuclei size = {int(np.pi*(diameter/10)**2)}')
    masks, flows, styles, diams = model.eval(dapi, diameter=diameter, channels=channels,flow_threshold=0,min_size= int(np.pi*(diameter/10)**2))
    return({newkey:masks})

def cellpose_cell(key,zdh,diameter=25):
    '''
    big tumor cell is 30 pixels, lymphocyte about 18 pixels, small fibroblast 12 pixels
    '''
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu())
        mx_gpu = mx.gpu()
    except mx.MXNetError:
        mx_gpu = None
    model = models.Cellpose(model_type='cyto',device=mx_gpu)
    newkey = f"{key.split(' - Z')[0]} cell{diameter}"
    print(f"modelling {newkey}")
    channels = [2,3]
    print(f'Minimum cell size = {int(np.pi*(diameter/5)**2)}')
    masks, flows, styles, diams = model.eval(zdh, diameter=diameter, channels=channels,flow_threshold=0.6,cellprob_threshold=0.0, min_size= int(np.pi*(diameter/5)**2))
    return({newkey:masks})

def parse_org(s_end = "ORG.tif",s_start='R'):
    """
    This function will parse images following koei's naming convention
    Example: Registered-R1_PCNA.CD8.PD1.CK19_Her2B-K157-Scene-002_c1_ORG.tif
    The output is a dataframe with image filename in index
    And rounds, color, imagetype, scene (/tissue), and marker in the columns
    """
    s_path = os.getcwd()
    ls_file = []
    for file in os.listdir():
        if file.endswith(s_end):
            if file.find(s_start)==0:
                ls_file = ls_file + [file]
    df_img = pd.DataFrame(index=ls_file)
    df_img['rounds'] = [item.split('_')[0].split('Registered-')[1] for item in df_img.index]
    df_img['color'] = [item.split('_')[-2] for item in df_img.index]
    df_img['slide'] = [item.split('_')[2] for item in df_img.index]
    df_img['marker_string'] = [item.split('_')[1] for item in df_img.index]
    try:
        df_img['scene'] = [item.split('-Scene-')[1] for item in df_img.slide]
    except:
        df_img['scene'] = '001'
    df_img['path'] = [f"{s_path}/{item}" for item in df_img.index]
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
        #these are only included in sardana shading corrected images
        elif s_color == 'c6':
            s_marker = s_index.split('_')[1].split('.')[2]
        elif s_color == 'c7':
            s_marker = s_index.split('_')[1].split('.')[3]
        else: print('Error')
        df_img.loc[s_index,'marker'] = s_marker
    return(df_img) 

def cmif_mkdir(ls_dir):
    '''
    check if directories existe. if not, make them
    '''
    for s_dir in ls_dir:
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)

def load_single(s_find, s_scene):
    '''
    load a single image containing the find strin, scale, return {filename:scaled image}
    '''
    d_img = {}
    for s_file in os.listdir():
        if s_file.find(s_find)>-1:
            a_img = io.imread(s_file)
            a_scale = skimage.exposure.rescale_intensity(a_img,in_range=(np.quantile(a_img,0.03),1.5*np.quantile(a_img,0.9999)))
            #d_img.update({f"{os.path.splitext(s_file)[0]}":a_scale})
            d_img.update({f"{s_scene}":a_scale})
    print(f'Number of images = {len(d_img)}')
    return(d_img)

def load_stack(df_img,s_find,s_scene,ls_markers,ls_rare):
    '''
    load an image stack in df_img, (df_img must have "path")
    scale, get mip, return {filename:mip}
    '''
    d_img = {}
    for s_file in os.listdir():
        if s_file.find(s_find)>-1:
            a_img = io.imread(s_file)
            dapi = skimage.exposure.rescale_intensity(a_img,in_range=(np.quantile(a_img,0.03),1.5*np.quantile(a_img,0.9999)))
 
    imgs = []
    #images
    df_common = df_img[df_img.marker.isin(ls_markers) & ~df_img.marker.isin(ls_rare)]
    df_rare =  df_img[df_img.marker.isin(ls_markers) & df_img.marker.isin(ls_rare)]
    for s_path in df_common.path:
        #print(s_path)
        img = io.imread(s_path)
        img_scale = skimage.exposure.rescale_intensity(img,in_range=(np.quantile(img,0.03),1.5*np.quantile(img,0.9999)))
        imgs.append(img_scale)
    for s_path in df_rare.path:
        img = io.imread(s_path)
        img_scale = skimage.exposure.rescale_intensity(img,in_range=(np.quantile(img,0.03),1.5*np.quantile(img,0.99999)))
        imgs.append(img_scale)
    mip = np.stack(imgs).max(axis=0)
    zdh = np.dstack((np.zeros(mip.shape),mip,dapi)).astype('uint16')
    #name
    #s_index = df_common.index[0]
    #s_common_marker = df_common.loc[s_index,'marker_string']
    #s_name = os.path.splitext(df_common.index[0])[0]
    #s_name = s_name.replace(s_common_marker,".".join(ls_markers))
    # name
    s_name = f'{s_scene}_{".".join(ls_markers)}'
    d_img.update({s_name:zdh})
    print(f'Number of projection images = ({len(d_img)}')
    return(d_img)

def load_img(subdir,s_find,s_sample,s_scene,ls_seg_markers,ls_rare):
    '''
    load dapi round and cell segmentation images
    '''
   #image dataframe
    os.chdir(subdir)
    df_seg = pd.DataFrame()
    for s_dir in os.listdir():
        if s_dir.find(s_sample)>-1:
            os.chdir(s_dir)
            df_img = parse_org()
            df_markers = df_img[df_img.marker.isin(ls_seg_markers)]
            df_markers['path'] = [f'{subdir}/{s_dir}/{item}' for item in df_markers.index]
            if df_img.index.str.contains(s_find).sum()==1:
                s_file = s_dir
                dapi = io.imread(df_img[df_img.index.str.contains(s_find)].index[0])
            os.chdir('..')
            df_seg = df_seg.append(df_markers)

    #load z_projection DAPIs
    os.chdir(subdir)
    d_dapi = {}
    d_cyto = {}

    dapi_scale = skimage.exposure.rescale_intensity(dapi,in_range=(np.quantile(dapi,0.03),1.5*np.quantile(dapi,0.9999)))
    d_dapi.update({f"{s_sample}-{s_scene}":dapi_scale})
    imgs = []
    #images
    df_common = df_seg[(df_seg.scene==s_scene) & (~df_seg.marker.isin(ls_rare))]
    df_rare =  df_seg[(df_seg.scene==s_scene) & (df_seg.marker.isin(ls_rare))]
    for s_path in df_common.path:
                print(s_path)
                img = io.imread(s_path)
                img_scale = skimage.exposure.rescale_intensity(img,in_range=(np.quantile(img,0.03),1.5*np.quantile(img,0.9999)))
                imgs.append(img_scale)
    for s_path in df_rare.path:
                img = io.imread(s_path)
                img_scale = skimage.exposure.rescale_intensity(img,in_range=(np.quantile(img,0.03),1.5*np.quantile(img,0.99999)))
                imgs.append(img_scale)
    mip = np.stack(imgs).max(axis=0)
    zdh = np.dstack((np.zeros(mip.shape),mip,dapi)).astype('uint16')
    d_cyto.update({f"{s_sample}-{s_scene}":zdh})
    print(f'Number of images = {len(d_dapi)} dapi projections ({len(d_cyto)} cytoplasm projections) ')

    return(d_dapi,d_cyto)

def cellpose_segment_job(s_sample='SampleName',s_slide_scene="SceneName",s_find="FindDAPIString",segdir='PathtoSegmentation',imgdir='PathtoImages',nuc_diam='30',cell_diam='30',s_type='cell_or_nuclei',s_seg_markers="['Ecad']",s_rare="[]",s_match='both',s_data='cmIF',s_job='cpu'):
    """
    makes specific changes to template pyscripts files in Jenny's directories to result in .py file
    Input:
    """
    #find template, open ,edit
    os.chdir(f'{s_src_path}/src')
    if s_data == 'cmIF':
        with open('cellpose_template.py') as f:
            s_file = f.read()
    elif s_data == 'codex':
        with open('cellpose_template_codex.py') as f:
            s_file = f.read()
    s_file = s_file.replace('SampleName',s_sample)
    s_file = s_file.replace('SceneName',s_slide_scene)
    s_file = s_file.replace('FindDAPIString',s_find)
    s_file = s_file.replace('nuc_diam=int',f'nuc_diam={str(nuc_diam)}')
    s_file = s_file.replace('cell_diam=int',f'cell_diam={str(cell_diam)}')
    s_file = s_file.replace('cell_or_nuclei',s_type)
    s_file = s_file.replace("['Ecad']",s_seg_markers)
    s_file = s_file.replace("ls_rare = []",f"ls_rare = {s_rare}")
    s_file = s_file.replace('PathtoSegmentation',segdir)
    s_file = s_file.replace('PathtoImages',imgdir)
    if s_match == 'match':
        s_file = s_file.replace('#MATCHONLY',"'''")
    elif s_match == 'seg':
        s_file = s_file.replace('#SEGONLY',"'''")
    if s_job == 'long':
        with open('cellpose_template_long.sh') as f:
            s_shell = f.read()
    elif s_job == 'gpu':
        with open('cellpose_template_gpu.sh') as f:
            s_shell = f.read()
        s_file = s_file.replace('#gpu#','')
        s_file = s_file.replace('#SEGONLY',"'''")
    else:
        with open('cellpose_template.sh') as f:
            s_shell = f.read()
    s_shell = s_shell.replace("PythonScripName",f'cellpose_{s_type}_{s_slide_scene}.py')

    #save edited .py file
    if s_sample.find("-Scene") > -1:
        s_sample = s_sample.split("-Scene")[0]
        print(s_sample)
    os.chdir(f'{segdir}')
    with open(f'cellpose_{s_type}_{s_slide_scene}.py', 'w') as f:
        f.write(s_file)

    with open(f'cellpose_{s_type}_{s_slide_scene}.sh', 'w') as f:
        f.write(s_shell)
    st = os.stat(f'cellpose_{s_type}_{s_slide_scene}.sh')
    os.chmod(f'cellpose_{s_type}_{s_slide_scene}.sh', st.st_mode | stat.S_IEXEC)

def segment_spawner(s_sample,segdir,regdir,nuc_diam=30,cell_diam=30,s_type='nuclei',s_seg_markers="['Ecad']",s_job='short',s_match='both'):
    '''
    spawns cellpose segmentation jobs by modifying a python and bash script, saving them and calling with os.system
    s_job='gpu' or 'long' (default = 'short')
    s_match= 'seg' or 'match' (default = 'both')
    '''
    preprocess.cmif_mkdir([f'{segdir}/{s_sample}Cellpose_Segmentation'])
    os.chdir(f'{regdir}')
    for s_file in os.listdir():
        if s_file.find(s_sample) > -1:
            os.chdir(f'{regdir}/{s_file}')
            print(f'Processing {s_file}')
            df_img = parse_org()
            for s_scene in sorted(set(df_img.scene)):
                s_slide_scene= f'{s_sample}-Scene-{s_scene}'
                s_find = df_img[(df_img.rounds=='R1') & (df_img.color=='c1') & (df_img.scene==s_scene)].index[0]
                if os.path.exists(f'{regdir}/{s_slide_scene}'):
                    cellpose_segment_job(s_file,s_slide_scene,s_find,f'{segdir}/{s_sample}Cellpose_Segmentation',f'{regdir}/{s_slide_scene}',nuc_diam,cell_diam,s_type,s_seg_markers,s_job=s_job, s_match=s_match)
                elif os.path.exists(f'{regdir}/{s_sample}'):
                    cellpose_segment_job(s_file,s_slide_scene,s_find,f'{segdir}/{s_sample}Cellpose_Segmentation',f'{regdir}/{s_sample}',nuc_diam,cell_diam,s_type,s_seg_markers,s_job=s_job, s_match=s_match)
                os.chdir(f'{segdir}/{s_sample}Cellpose_Segmentation')
                os.system(f'sbatch cellpose_{s_type}_{s_slide_scene}.sh')
                time.sleep(4)
                print('Next')

def save_seg(processed_list,segdir,s_type='nuclei'):
    '''
    save the segmentation basins
    '''

    for item in processed_list:
        for newkey,mask in item.items():
            print(f"saving {newkey.split(' - ')[0]} {s_type} Basins")
            if s_type=='nuclei':
                io.imsave(f"{segdir}/{newkey} - Nuclei Segmentation Basins.tif", mask) #Scene 002 - Nuclei Segmentation Basins.tif
            elif s_type=='cell':
                io.imsave(f"{segdir}/{newkey} - Cell Segmentation Basins.tif", mask) #Scene 002 - Nuclei Segmentation Basins.tif

def save_img(d_img, segdir,s_type='nuclei',ls_seg_markers=[]):
    '''
    save the segmentation basins
    '''
    #save dapi or save the cyto projection
    if s_type=='nuclei':
        for key,dapi in d_img.items():
            print('saving DAPI')
            print(key)
            io.imsave(f"{segdir}/{key} - DAPI.png",dapi)
    elif s_type=='cell':
        for key,zdh in d_img.items():
            print('saving Cyto Projection')
            io.imsave(f"{segdir}/{key.split(' - ')[0]} - {'.'.join(ls_seg_markers)}_CytoProj.png",(zdh/255).astype('uint8'))

    else:
        print('choose nuceli or cell')

# numba functions
kv_ty = (types.int64, types.int64)

@jitclass([('d', types.DictType(*kv_ty)),
           ('l', types.ListType(types.float64))])
class ContainerHolder(object):
    def __init__(self):
        # initialize the containers
        self.d = numba.typed.Dict.empty(*kv_ty)
        self.l = numba.typed.List.empty_list(types.float64)

@overload(np.array)
def np_array_ol(x):
    if isinstance(x, types.Array):
        def impl(x):
            return np.copy(x)
        return impl

@numba.njit
def test(a):
    b = np.array(a)

# numba function
    '''
    use numba to quickly iterate over each label and replace pixels with new pixel values
    Input:
    container = numba container class, with key-value pairs of old-new cell IDs
    labels: numpy array with labels to rename
        #cell_labels = np.where(np.array(cell_labels,dtype=np.int64)==key, value, np.array(labels,dtype=np.int64))
    '''

@jit(nopython=True)
def relabel_numba(container,cell_labels):
    '''
    faster; replace pixels accorind to dictionsry (i.e. numba container)
    key is original cell label, value is replaced label
    '''
    cell_labels = np.array(cell_labels)
    for key, value in container.d.items():
        cell_labels = np.where(cell_labels==key, value, cell_labels)
    print('done matching')
    return(cell_labels)

def relabel_numpy(d_replace,cell_labels):
    '''
    slow replace pixels accorind to dictionary 
    key is original cell label, value is replaced label
    '''
    #key is original cell albel, value is replaced label
    for key, value in d_replace.items():
        cell_labels = np.where(cell_labels==key, value, cell_labels)
    print('done matching')
    return(cell_labels)

def relabel_gpu(d_replace,cell_labels):
    '''
    not implemented yet
    key is original cell label, value is replaced label
    '''
    #key is original cell albel, value is replaced label
    for key, value in d_replace.items():
        cell_labels = np.where(cell_labels==key, value, cell_labels)
    print('done mathcing')
    return(cell_labels)

def nuc_to_cell_new(labels,cell_labels):
    '''
    problem - still not giving same result as original function
    associate the largest nucleaus contained in each cell segmentation
    Input:
    labels: nuclear labels
    cell_labels: cell labels that need to be matched
    Ouput:
    container: numba container of key-value pairs of old-new cell IDs
    '''
    start = time.time()
    #dominant nuclei
    props = measure.regionprops_table(cell_labels,labels, properties=(['intensity_image','image','label']))
    df_prop = pd.DataFrame(props)
    d_replace = {}
    for idx in df_prop.index[::-1]:
        label_id = df_prop.loc[idx,'label']
        intensity_image = df_prop.loc[idx,'intensity_image']
        image = df_prop.loc[idx,'image']
        nuc_labels = intensity_image[image & intensity_image!=0]
        if len(nuc_labels) == 0:
            d_replace.update({label_id:0}) 
        elif len(np.unique(nuc_labels)) == 1:
            d_replace.update({label_id:nuc_labels[0]})
        else:
            new_id = scipy.stats.mode(nuc_labels)[0][0]
            d_replace.update({label_id:new_id})

    #convert to numba container
    container = ContainerHolder()
    for key, value in d_replace.items():
        container.d[key] = value
    end = time.time()
    print(end - start)
    return(container,d_replace, df_prop) 

def nuc_to_cell(labels,cell_labels):
    '''
    associate the largest nucleaus contained in each cell segmentation
    Input:
    labels: nuclear labels
    cell_labels: cell labels that need to be matched
    Ouput:
    container: numba container of key-value pairs of old-new cell IDs
    '''
    start = time.time()
    #dominant nuclei
    d_replace = {}
    for idx in np.unique(cell_labels)[::-1]:
        if idx == 0:
            continue
        #iterate over each cell label, find all non-zero values contained within that mask
        cell_array = labels[cell_labels == idx]
        cell_array =cell_array[cell_array !=0]
        #for multiple nuclei, choose largest (most common pixels, i.e. mode)
        if len(np.unique(cell_array)) > 1:
            new_id = scipy.stats.mode(cell_array, axis=0)[0][0]
            d_replace.update({idx:new_id})
        elif len(np.unique(cell_array)) == 1:
            d_replace.update({idx:cell_array[0]})
        else:
            d_replace.update({idx:0})
    #fix matching bug
    d_replace = {item[0]:item[1] for item in sorted(d_replace.items(), key=lambda x: x[1], reverse=True)}
    #convert to numba container
    container = ContainerHolder()
    for key, value in d_replace.items():
        container.d[key] = value
    end = time.time()
    print(end - start)
    return(container,d_replace)

########## OLD ##############

def zero_background(cells_relabel):
    '''
    in a labelled cell image, set the background to zero
    '''
    mode = stats.mode(cells_relabel,axis=0)[0][0][0]
    black = cells_relabel.copy()
    black[black==mode] = 0
    return(black)

def nuc_to_cell_watershed(labels,cell_labels,i_small=200):
    '''
    associate the largest nucleus contained in each cell segmentation
    Input:
    labels: nuclear labels
    cell_labels: cell labels that need to be matched
    Ouput:
    new_cell_labels: shrunk so not touching and cleaned of small objects < i_small
    container: numba container of key-value pairs of old-new cell IDs
    d_replace: python dictionary of key-value pairs
    '''
    #cells
    cell_boundaries = segmentation.find_boundaries(cell_labels,mode='outer')
    shrunk_cells = cell_labels.copy()
    shrunk_cells[cell_boundaries] = 0
    foreground = shrunk_cells != 0
    foreground_cleaned = morphology.remove_small_objects(foreground, i_small)
    background = ~foreground_cleaned
    shrunk_cells[background] = 0
    #problem when we filter
    #new_cell_labels = measure.label(foreground_cleaned, background=0)

    #nuclei
    cut_labels = labels.copy()
    background = ~foreground_cleaned
    cut_labels[background] = 0
    labels_in = morphology.remove_small_objects(cut_labels, i_small)
    cleaned_nuclei = labels_in
    distance = ndi.distance_transform_edt(foreground_cleaned)
    labels_out = segmentation.watershed(-distance, labels_in, mask=foreground_cleaned)

    #dominant nuclei
    props = measure.regionprops_table(shrunk_cells,labels_out, properties=('min_intensity','max_intensity','mean_intensity'))
    df_prop = pd.DataFrame(props)
    d_replace = {}
    for idx in df_prop.index[::-1]:
        #iterate over each cell label, find all non-zero values of watershed expansioncontained within that mask 
        cell_array = labels_out[shrunk_cells == idx]
        if len(np.unique(cell_array)) > 1:
            new_id = scipy.stats.mode(cell_array, axis=0)[0][0]
            d_replace.update({idx:new_id})
        elif len(np.unique(cell_array)) == 1:
            d_replace.update({idx:cell_array[0]})
        else:
            d_replace.update({idx:0})
    #convert to numba container
    container = ContainerHolder()
    for key, value in d_replace.items():
        container.d[key] = value

    return(container)

def save_seg_z(processed_list,segdir,s_type='nuclei'):
    '''
    save the segmentation basins
    '''

    for item in processed_list:
        for newkey,mask in item.items():
            print(f"saving {newkey.split(' - Z')[0]} {s_type} Basins")
            if s_type=='nuclei':
                io.imsave(f"{segdir}/{newkey} - Nuclei Segmentation Basins.tif", mask) #Scene 002 - Nuclei Segmentation Basins.tif
            elif s_type=='cell':
                io.imsave(f"{segdir}/{newkey} - Cell Segmentation Basins.tif", mask) #Scene 002 - Nuclei Segmentation Basins.tif

def cellpose_segment_parallel(d_img,s_type='nuclei'):
    '''
    Dont use/ segment nuclei or cell
    '''
    if s_type=='nuclei':
        print('segmenting nuclei')
        if __name__ == "__main__":
            processed_list = Parallel(n_jobs=len(d_img))(delayed(cellpose_nuc)(key,img,diameter=nuc_diam) for key,img in d_img.items())

    elif s_type=='cell':
        print('segmenting cells')
        if __name__ == "__main__":
            processed_list = Parallel(n_jobs=len(d_img))(delayed(cellpose_cell)(key,img,diameter=cell_diam) for key,img in d_img.items())

    else:
        print('choose nuceli or cell')
    return(processed_list)

def save_img_z(d_img, segdir,s_type='nuclei',ls_seg_markers=[]):
    '''
    save the segmentation basins
    '''
    #save dapi or save the cyto projection
    if s_type=='nuclei':
        for key,dapi in d_img.items():
            print('saving DAPI')
            io.imsave(f"{segdir}/{key}",dapi)
    elif s_type=='cell':
        for key,zdh in d_img.items():
            print('saving Cyto Projection')
            io.imsave(f"{segdir}/{key.split(' - Z')[0]} - {'.'.join(ls_seg_markers)}_CytoProj.png",(zdh/255).astype('uint8'))

    else:
        print('choose nuceli or cell')

def cellpose_segment_job_z(s_sample='SampleName',s_scene="SceneName",nuc_diam='20',cell_diam='25',s_type='cell_or_nuclei',s_seg_markers="['Ecad']",s_rare="[]",codedir='PathtoCode'):
    """
    makes specific changes to template pyscripts files in Jenny's directories to result in .py file
    Input:

    """
    #find template, open ,edit
    os.chdir(f'{s_src_path}/src')
    with open('cellpose_template_z.py') as f:
            s_file = f.read()
    s_file = s_file.replace('SampleName',s_sample)
    s_file = s_file.replace('SceneName',s_scene)
    s_file = s_file.replace('nuc_diam=int',f'nuc_diam={str(nuc_diam)}')
    s_file = s_file.replace('cell_diam=int',f'cell_diam={str(cell_diam)}')
    s_file = s_file.replace('cell_or_nuclei',s_type)
    s_file = s_file.replace("['Ecad']",s_seg_markers)
    s_file = s_file.replace("ls_rare = []",f"ls_rare = {s_rare}")
    s_file = s_file.replace('PathtoCode',codedir)

    with open('cellpose_template_z.sh') as f:
        s_shell = f.read()
        s_shell = s_shell.replace("PythonScripName",f'cellpose_{s_type}_{s_scene.replace(" ","-").split("_")[0]}.py')

    #save edited .py file
    os.chdir(f'{codedir}/Segmentation/{s_sample}Cellpose_Segmentation')
    with open(f'cellpose_{s_type}_{s_scene.replace(" ","-").split("_")[0]}.py', 'w') as f:
        f.write(s_file)

    with open(f'cellpose_{s_type}_{s_scene.replace(" ","-").split("_")[0]}.sh', 'w') as f:
        f.write(s_shell)

def load_scene_z(subdir,dapidir,s_sample,s_scene,ls_seg_markers,ls_rare):
    '''
    load dapi projection and cell segmentation images
    '''
   #image dataframe
    os.chdir(subdir)
    df_seg = pd.DataFrame()
    for s_dir in os.listdir():
        if s_dir.find(s_sample)>-1:
            os.chdir(s_dir)
            df_img = parse_org()
            df_markers = df_img[df_img.marker.isin(ls_seg_markers)]
            df_markers['path'] = [f'{subdir}/{s_dir}/{item}' for item in df_markers.index]
            os.chdir('..')
            df_seg = df_seg.append(df_markers)

    #load z_projection DAPIs
    os.chdir(dapidir)
    d_dapi = {}
    d_cyto = {}
    for s_file in sorted(os.listdir()):
        #print(s_file)
        if s_file.find(f'{s_scene} - ZProjectionDAPI.png')>-1:
            dapi = io.imread(s_file)
            dapi_scale = skimage.exposure.rescale_intensity(dapi,in_range=(np.quantile(dapi,0.03),1.5*np.quantile(dapi,0.9999)))
            d_dapi.update({s_file:dapi_scale})
            s_scene = s_scene.split(' ')[1].split('_')[0]
            print(s_scene)
            imgs = []
            #images
            df_common = df_seg[(df_seg.scene==s_scene) & (~df_markers.marker.isin(ls_rare))]
            df_rare =  df_seg[(df_seg.scene==s_scene) & (df_markers.marker.isin(ls_rare))]
            for s_path in df_common.path:
                img = io.imread(s_path)
                img_scale = skimage.exposure.rescale_intensity(img,in_range=(np.quantile(img,0.03),1.5*np.quantile(img,0.9999)))
                imgs.append(img_scale)
            for s_path in df_rare.path:
                img = io.imread(s_path)
                img_scale = skimage.exposure.rescale_intensity(img,in_range=(np.quantile(img,0.03),1.5*np.quantile(img,0.999999)))
                imgs.append(img_scale)
            mip = np.stack(imgs).max(axis=0)
            zdh = np.dstack((np.zeros(mip.shape),mip,dapi)).astype('uint16')
            d_cyto.update({s_file:zdh})
    print(f'Number of images = {len(d_dapi)} dapi projections ({len(d_cyto)} cytoplasm projections) ')

    return(d_dapi,d_cyto)

#test code
'''
import napari
#os.chdir('./Desktop/BR1506')
labels = io.imread('Scene 059 nuclei20 - Nuclei Segmentation Basins.tif')
cell_labels = io.imread('Scene 059 cell25 - Cell Segmentation Basins.tif')
cyto_img = io.imread('Scene 059 - CytoProj.png')
dapi_img = io.imread('Scene 059 - ZProjectionDAPI.png')
viewer = napari.Viewer()
viewer.add_labels(labels,blending='additive')
viewer.add_labels(cell_labels,blending='additive')
viewer.add_image(cyto_img,blending='additive')
viewer.add_image(dapi_img,blending='additive',colormap='blue')
#cell_boundaries = segmentation.find_boundaries(cell_labels,mode='outer')
#viewer.add_labels(cell_boundaries,blending='additive')
#nuclear_boundaries = segmentation.find_boundaries(labels,mode='outer')
#viewer.add_labels(nuclear_boundaries,blending='additive',num_colors=2)
closing = skimage.morphology.closing(cell_labels)
viewer.add_labels(closing,blending='additive')
container = nuc_to_cell(labels,closing)#cell_labels)

#matched cell labels
cells_relabel = relabel_numba(container[0],closing)
#remove background
mode = stats.mode(cells_relabel,axis=0)[0][0][0]
black = cells_relabel.copy()
black[black==mode] = 0
viewer.add_labels(black,blending='additive')
cell_boundaries = segmentation.find_boundaries(cells_relabel,mode='outer')
viewer.add_labels(cell_boundaries,blending='additive')
#ring
overlap = black==labels
viewer.add_labels(overlap, blending='additive')
#cytoplasm
ring_rep = black.copy()
ring_rep[overlap] = 0
viewer.add_labels(ring_rep, blending='additive')
#membrane
rim_labels = contract_membrane(black)
viewer.add_labels(rim_labels, blending='additive')

#expanded nucleus
__,__,peri_nuc = expand_nuc(labels,distance=3)
viewer.add_labels(peri_nuc, blending='additive')
'''