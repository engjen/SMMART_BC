#####
# gating.py
# author:  engje, grael
# date: 2020-04-07
# license: GPLv3
#####

# library
import os
import pandas as pd
import shutil
from mplex_image import analyze
import numpy as np


def main_celltypes(df_data,ls_endothelial,ls_immune,ls_tumor,ls_cellline_index):
    #celltpye
    #1 endothelial
    df_data['endothelial'] = df_data.loc[:,ls_endothelial].any(axis=1)
    #2 immune
    ls_exclude = ls_endothelial 
    df_data['immune'] = df_data.loc[:,ls_immune].any(axis=1) & ~df_data.loc[:,ls_exclude].any(axis=1)
    #3 tumor
    ls_exclude =  ls_endothelial + ls_immune
    df_data['tumor'] = df_data.loc[:,ls_tumor].any(axis=1) & ~df_data.loc[:,ls_exclude].any(axis=1) 
    #4 stromal
    ls_exclude = ls_immune + ls_endothelial + ls_tumor
    df_data['stromal'] = ~df_data.loc[:,ls_exclude].any(axis=1)
    #add celltype
    ls_cell_names = ['stromal','endothelial','tumor','immune']
    s_type_name = 'celltype'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    #fix cell lines (all tumor!)
    df_data['slide_scene'] = [item.split('_cell')[0] for item in df_data.index]
    df_data.loc[df_data[df_data.slide_scene.isin(ls_cellline_index)].index,'celltype'] = 'tumor'
    df_data['immune'] = df_data.loc[:,'celltype'] == 'immune'
    df_data['stromal'] = df_data.loc[:,'celltype'] == 'stromal'
    df_data['endothelial'] = df_data.loc[:,'celltype'] == 'endothelial'
    return(df_data)

def proliferation(df_data,ls_prolif):
    #proliferation
    df_data['prolif'] = df_data.loc[:,ls_prolif].any(axis=1)
    df_data['nonprolif'] = ~df_data.loc[:,ls_prolif].any(axis=1)
    #add proliferation
    ls_cell_names = ['prolif','nonprolif']
    s_type_name = 'proliferation'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    return(df_data)

def immune_types(df_data,s_myeloid,s_bcell,s_tcell):
    ## T cell, B cell or myeloid
    df_data['CD68Mac'] = df_data.loc[:,[s_myeloid,'immune']].all(axis=1) 
    df_data['CD20Bcell'] = df_data.loc[:,[s_bcell,'immune']].all(axis=1) & ~df_data.loc[:,['CD68Mac',s_tcell]].any(axis=1)
    df_data['TcellImmune'] = df_data.loc[:,[s_tcell,'immune']].all(axis=1) & ~df_data.loc[:,['CD20Bcell','CD68Mac']].any(axis=1)
    df_data['UnspecifiedImmune'] = df_data.loc[:,'immune'] & ~df_data.loc[:,['CD20Bcell','TcellImmune','CD68Mac']].any(axis=1)
    ## CD4 and CD8 
    if df_data.columns.isin(['CD8_Ring','CD4_Ring']).sum()==2:
        #print('CD4 AND CD8')
        df_data['CD8Tcell'] = df_data.loc[: ,['CD8_Ring','TcellImmune']].all(axis=1)
        df_data['CD4Tcell'] = df_data.loc[: ,['CD4_Ring','TcellImmune']].all(axis=1) & ~df_data.loc[: ,'CD8Tcell']
        df_data['UnspecifiedTcell'] = df_data.TcellImmune & ~df_data.loc[:,['CD8Tcell','CD4Tcell']].any(axis=1) #if cd4 or 8 then sum = 2
        ## check
        ls_immune = df_data[df_data.loc[:,'TcellImmune']].index.tolist()
        if ((df_data.loc[ls_immune,['CD8Tcell','CD4Tcell','UnspecifiedTcell']].sum(axis=1)!=1)).any():
            print('Error in Tcell cell types')
        ls_immuntype = ['CD68Mac','CD20Bcell','UnspecifiedImmune','CD8Tcell','CD4Tcell','UnspecifiedTcell'] #'TcellImmune',
    #add Immunetype
    ls_cell_names = ls_immuntype
    s_type_name = 'ImmuneType'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)

    #get rid of unspecfied immune cells (make them stroma)
    ls_index = df_data[df_data.ImmuneType.fillna('x').str.contains('Unspecified')].index
    df_data.loc[ls_index,'celltype'] = 'stromal'
    df_data.loc[ls_index,'ImmuneType'] = np.nan
    df_data.loc[ls_index,'stromal'] = True
    df_data.loc[ls_index,'immune'] = False
    return(df_data)

def immune_functional(df_data,ls_immune_functional):
    #Immune functional states 
    df_data.rename(dict(zip(ls_immune_functional,[item.split('_')[0] for item in ls_immune_functional])),axis=1,inplace=True)
    df_func = analyze.combinations(df_data,[item.split('_')[0] for item in ls_immune_functional])
    df_data = df_data.merge(df_func,how='left', left_index=True, right_index=True, suffixes = ('_all',''))
    #gated combinations: immune type plus fuctional status
    ls_gate = sorted(df_data[~df_data.ImmuneType.isna()].loc[:,'ImmuneType'].unique())
    ls_marker = df_func.columns.tolist()
    df_gate_counts = analyze.gated_combinations(df_data,ls_gate,ls_marker)
    df_data = df_data.merge(df_gate_counts, how='left', left_index=True, right_index=True,suffixes = ('_all',''))
    #add FuncImmune
    ls_cell_names = df_gate_counts.columns.tolist()
    s_type_name ='FuncImmune'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    return(df_data)

########################################
#CellProlif combinations, main cell types and proliferation
######################################
def cell_prolif(df_data, s_gate='celltype',ls_combo =['prolif','nonprolif']):
    ls_gate = df_data.loc[:,s_gate].unique().tolist()
    df_gate_counts2 = analyze.gated_combinations(df_data,ls_gate,ls_combo)
    df_data = df_data.merge(df_gate_counts2, how='left', left_index=True, right_index=True,suffixes = ('_all',''))
    #add CellProlif
    ls_cell_names = ['endothelial_prolif','endothelial_nonprolif', 'tumor_prolif', 'tumor_nonprolif',
       'stromal_prolif', 'stromal_nonprolif', 'immune_prolif','immune_nonprolif']
    ls_cell_names = df_gate_counts2.columns.tolist()
    s_type_name = 'CellProlif'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    return(df_data)

def diff_hr_state(df_data,ls_luminal,ls_basal,ls_mes):
    ls_mes = df_data.columns[(df_data.dtypes=='bool') & (df_data.columns.isin(ls_mes) | df_data.columns.isin([item.split('_')[0] for item in ls_mes]))].tolist()
    print('differentiation')
    df_data['Lum'] = df_data.loc[:,ls_luminal].any(axis=1) & df_data.tumor
    df_data['Bas'] = df_data.loc[:,ls_basal].any(axis=1)  & df_data.tumor
    df_data['Mes'] = df_data.loc[:,ls_mes].any(axis=1) & df_data.tumor

    print('hormonal status')
    df_data['ER'] = df_data.loc[:,['tumor','ER_Nuclei']].all(axis=1)
    df_data['HER2'] = df_data.loc[:,['tumor','HER2_Ring']].all(axis=1)
    ls_hr = ['ER']
    if df_data.columns.isin(['PgR_Nuclei']).any():
        df_data['PR'] = df_data.loc[:,['tumor','PgR_Nuclei']].all(axis=1)
        ls_hr.append('PR')

    df_data['HR'] = df_data.loc[:,ls_hr].any(axis=1) & df_data.tumor

    ls_marker = ['Lum','Bas','Mes'] #
    df_diff = analyze.combinations(df_data,ls_marker)
    df_data = df_data.merge(df_diff,how='left', left_index=True, right_index=True, suffixes = ('_all',''))

    #add DiffState
    ls_cell_names = df_diff.columns.tolist()
    s_type_name = 'DiffState'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    #change non-tumor to NA (works!)
    df_data.loc[df_data[df_data.celltype != 'tumor'].index,s_type_name] = np.nan

    #2 ER/PR/HER2
    ls_marker =  ['HR','HER2']
    df_hr = analyze.combinations(df_data,ls_marker)
    df_hr.rename({'__':'TN'},axis=1,inplace=True)
    df_data = df_data.merge(df_hr,how='left', left_index=True, right_index=True,suffixes = ('_all',''))
    ls_cell_names = df_hr.columns.tolist()
    s_type_name = 'HRStatus'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    #change non-tumor to NA (works!)
    df_data.loc[df_data[df_data.celltype != 'tumor'].index,s_type_name] = np.nan

    #3 combinations: differentiation and HR status
    ls_gate = df_diff.columns.tolist()
    ls_marker = df_hr.columns.tolist()
    df_gate_counts = analyze.gated_combinations(df_data,ls_gate,ls_marker)
    df_data = df_data.merge(df_gate_counts, how='left', left_index=True, right_index=True,suffixes = ('_all',''))

    # make Tumor Diff plus HR Status object column
    ls_cell_names =  df_gate_counts.columns.tolist()
    s_type_name = 'DiffStateHRStatus'
    analyze.add_celltype(df_data, ls_cell_names, s_type_name)
    #change non-tumor to NA (works!)
    df_data.loc[df_data[df_data.celltype != 'tumor'].index,s_type_name] = np.nan
    return(df_data)

def celltype_gates(df_data,ls_gate,s_new_name,s_celltype):
    '''
    multipurpose for stromaTumor
    ls_gates = 
    '''
    ls_gate = df_data.columns[(df_data.dtypes=='bool') & (df_data.columns.isin(ls_gate) | df_data.columns.isin([item.split('_')[0] for item in ls_gate]))].tolist()
    #tumor signaling and proliferation
    #rename
    df_data.rename(dict(zip(ls_gate,[item.split('_')[0] for item in ls_gate])),axis=1,inplace=True)
    ls_marker = [item.split('_')[0] for item in ls_gate]
    #functional states (stromal) (don't forget to merge!)
    df_func = analyze.combinations(df_data,ls_marker)
    df_data = df_data.merge(df_func,how='left', left_index=True, right_index=True, suffixes = ('_all',''))
    ls_cell_names = df_func.columns.tolist()
    analyze.add_celltype(df_data, ls_cell_names, s_new_name)
    #change non-tumor to NA (works!)
    df_data.loc[df_data[df_data.celltype != s_celltype].index,s_new_name] = np.nan
    df_data[s_new_name] = df_data.loc[:,s_new_name].replace(dict(zip(ls_cell_names,[f'{s_celltype}_{item}' for item in ls_cell_names])))
    return(df_data)

def non_tumor(df_data):
    #one more column: all non-tumor cells
    index_endothelial = df_data[df_data.celltype=='endothelial'].index
    index_immune = df_data[df_data.celltype=='immune'].index
    index_stroma = df_data[df_data.celltype=='stromal'].index
    index_tumor = df_data[df_data.celltype=='tumor'].index

    if df_data.columns.isin(['ImmuneType','StromalType']).sum() == 2:
        #fewer cell tpyes
        df_data.loc[index_endothelial,'NonTumor'] = 'endothelial'
        df_data.loc[index_immune,'NonTumor'] = df_data.loc[index_immune,'ImmuneType']
        df_data.loc[index_stroma,'NonTumor'] = df_data.loc[index_stroma,'StromalType']
        df_data.loc[index_tumor,'NonTumor'] = np.nan

        if df_data.columns.isin(['FuncImmune','CellProlif']).sum() == 2:
            #more cell types
            df_data.loc[index_endothelial,'NonTumorFunc'] = df_data.loc[index_endothelial,'CellProlif']
            df_data.loc[index_immune,'NonTumorFunc'] = df_data.loc[index_immune,'FuncImmune']
            df_data.loc[index_stroma,'NonTumorFunc'] = df_data.loc[index_stroma,'StromalType']
            df_data.loc[index_tumor,'NonTumorFunc'] = np.nan
    return(df_data)
