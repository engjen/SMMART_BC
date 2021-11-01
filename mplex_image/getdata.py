####
# title: getdata.py
#
# language: Python3.6
# date: 2018-08-00
# license: GPL>=v3
# author: Jenny, bue (mostly bue)
#
# description:
#   python3 library to analyise guillaume segemented cyclic staining data.
####

# load library
import csv
import os
import re


# function implementaion
# import importlib
# importlib.reload(getdata)

def get_df(
        #s_gseg_folder_root='/graylab/share/engje/Data/',
        #s_scene_label='Registered-Her'
        s_folder_regex="^SlideName.*_Features$",
        es_value_label = {"MeanIntensity","CentroidX","CentroidY"},
        #s_df_folder_root="./",
        #b_roundscycles=False,
    ):
    '''
    input:
        segmentation fiels from Guillaume's software, which have in the
        "Label" column the "cell serial number" (cell)
        and in other columns the "feature of intrests" and unintrest.

        the segmentation files are ordered in such a path structure:
        + {s_gseg_folder_root}
            |+ {s_gseg_folder_run_regex}*_YYYY-MM-DD_*  (run)
            |    |+ Scene 000 - Nuclei - CD32.txt (scene and protein)
            |    |+ Scene 000 - Location - ProteinName.txt
            |
            |+ {s_gseg_folder_run_regex}*_YYYY-MM-DD_*

    output:
        at {s_df_folder_root} tab separated value dataframe files
        per run and feature of intrest.
        y-axis: protein_location
        x-axis: scene_cell
        + runYYYYMMDD_MeanIntensity.tsv
        + runYYYYMMDD_{s_gseg_feature_label}.tsv

    run:
        import getdata
        getdata.get_df(s_gseg_folder_root='ihcData', s_gseg_folder_run_regex='^BM-Her2N75')

    description:
        function to extrtact dataframe like files of features of intrest
        from segmentation files from guilaumes segmentation software.
    '''
    # enter the data path
    #os.chdir(s_gseg_folder_root)
    
    
    # for each value label of intrest (such as MeanIntensity)
    for s_value_label in es_value_label:

        # for each run (such as folder BM-Her2N75-15_2017-08-07_Features)
        # change re.search to somehow specify folder of interest
        for s_dir in os.listdir():
            if re.search(s_folder_regex, s_dir):
                print(f"\nprocess {s_value_label} run: {s_dir}")
                # enter the run directory
                os.chdir(s_dir)
                # extract run label from dir name
                s_run = f"features_{s_dir.split('_')[0]}"
                # get empty run dictionary
                dd_run = {}

                # for each data file
                for s_file in os.listdir():
                    if re.search("^Scene", s_file):
                        print(f"process {s_value_label} file: {s_file} ...")
                        # extract scene from file name
                        ls_file = [s_splinter.strip() for s_splinter in s_file.split("-")] 
                        s_scene = re.sub("[^0-9a-zA-Z]", "", ls_file[0].lower()) #take out any alpha numberic 
                        # extract protein from file name
                        if (len(ls_file) < 3):
                            s_protein = f"{ls_file[1].split('.')[0]}" # this is dapi
                        else:
                            s_protein = f"{ls_file[2].split('.')[0]}_{ls_file[1]}" # others

                        # for each datarow in file
                        b_header = False  # header row inside file not yet found, so set flag false
                        with open(s_file, newline='') as f_csv:
                            o_reader = csv.reader(f_csv, delimiter=' ', quotechar='"')
                            for ls_row in o_reader:
                                if (b_header):
                                    # extract  cell label and data vale
                                    s_cell = ls_row[i_xcell]
                                    s_cell = f"{'0'*(5 - len(s_cell))}{s_cell}"
                                    o_value = ls_row[i_xvalue]
                                    # update run dictionary via scene_cell dictionery (one scene_cell dictionary per dataframe row)
                                    s_scene_cell = f"{s_scene}_cell{s_cell}"
                                    try:
                                        d_scene_cell = dd_run[s_scene_cell]  # we have already some data from this scene_cell
                                    except KeyError:
                                        d_scene_cell = {}  # this is the first time we deal with this scene_cell
                                    # update scene_cell dictionary with data values (one value inside dataframe row)
                                    try:
                                        o_there = d_scene_cell[s_protein]
                                        sys.exit(f"Error @ getDataframe : in run {s_run} code tries to populate dataframe row {s_scene_cell} column {s_protein} with a secound time (there:{o_there} new:{o_value}). this should never happen. code is messed up.")
                                    except KeyError:
                                        d_scene_cell.update({s_protein: o_value})
                                        dd_run.update({s_scene_cell: d_scene_cell})
                                else:
                                    #  extract cell label and data value of intrest column position
                                    i_xcell = ls_row.index("Label")
                                    i_xvalue = ls_row.index(s_value_label)
                                    b_header = True # header row found and information extracted, so set flag True

                # write run dictionar of dictionary into dataframe like file
                b_header = False
                s_file_output = f"../{s_run}_{s_value_label}.tsv"
                print(f"write file: {s_file_output}")
                with open(s_file_output, 'w', newline='') as f:
                    for s_scene_cell in sorted(dd_run):
                        ls_datarow = [s_scene_cell]
                        # handle protein column label row
                        if not (b_header):
                            ls_protein = sorted(dd_run[s_scene_cell])
                            print(ls_protein)
                            f.write("\t" + "\t".join(ls_protein) + "\n")
                            b_header = True
                        # handle data row
                        for s_protein in ls_protein:
                            o_value = dd_run[s_scene_cell][s_protein]
                            ls_datarow.append(o_value)
                        f.write("\t".join(ls_datarow) + "\n")
                        # sanity check
                        if (len(ls_protein) != (len(ls_datarow) -1)):
                            sys.exit(f"Error @ getDataframe : at {s_scene_cell} there are {len(ls_datarow) - len(ls_protein) -1} more proteins then in the aready writen rows")

                # jump back to the data path
                os.chdir("..")

    return(dd_run)


def dfextract(df_origin, s_extract, axis=0):
    '''
    input:
        df_origin: dataframe
        s_extract: index or column marker to be extacted
        axis: 0 specifies index to be extracted,
          1 specifies columns to be extracted

    output:
        df_extract: extracted dataframe

    run:
        import cycnorm
        cycnorm.dfyextract(df_scene, s_extract='CD74')
        cycnorm.dfextract(df_run, s_scene='scene86')

    description:
        function can extract e.g.
        specific scene datafarme from gseg2df generated run datafarme or
        specific protein from a scene dataframe.
    '''
    if (axis == 0):
        df_extract = df_origin.loc[df_origin.index.str.contains(s_extract),:]
    else:
        df_extract = df_origin.loc[:,df_origin.columns.str.contains(s_extract)]
    # output
    return(df_extract)
