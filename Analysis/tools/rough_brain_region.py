# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:20:40 2022

@author: ChuLab
"""

import pandas as pd
import csv
import numpy as np
import os

def get_structure_list(df):    
    structure_list = []
    for i in df['name'].index:
        structure_list.append(str(df["name"][i]))
    return structure_list


def create_new_df(structure_df, id_path):
    df = structure_df[structure_df["structure_id_path"].str.contains(id_path)]
    return df


def get_group_cell_density(structure_list, cfos_df):
    cell_number_list = []
    brain_region_volume_list = []
    for structure in structure_list:
        for i in range(cfos_df.shape[0]):
            if cfos_df["structure_name"][i]==structure :
                cell_number = float(cfos_df['total_cells'][i])
                brain_region_volume = float(cfos_df['total_volume_mm3'][i])
                cell_number_list.append(cell_number)
                brain_region_volume_list.append(brain_region_volume)
    cell_density = (np.sum(cell_number_list)/np.sum(brain_region_volume_list))
    try:
        cell_density = cell_density+1-1
    except:
        cell_density = 0
    return cell_density


state_csv_path = input("Insert the state csv file:\n") 
cfos_csv_path = input("INSERT THE FILE PATH:\n")
structure_csv = input("INSERT THE STRUCTURE FILE:\n")
savepath = input("INSERT THE SAVE PATH:\n")

state_df = pd.read_csv(state_csv_path)
brain_region = state_df["a"]
id_path_list = state_df["b"]
brain_region_name = state_df["c"]
structure_df = pd.read_csv(structure_csv)

group_list = ["control", "dominant", "subordinate"]
for group in group_list:
    os.makedirs(savepath +"\\"+ group, exist_ok = True)
    cfos_df_list = os.listdir(cfos_csv_path + "\\" + group)
    for cfos_csv in cfos_df_list:
        cfos_df = pd.read_csv(cfos_csv_path + "\\" + group + "\\" + cfos_csv)
        
        cell_density_list = []
        for n in id_path_list:
            data_df = create_new_df(structure_df, n)
            structure_list = get_structure_list(data_df)
            cell_density = get_group_cell_density(structure_list, cfos_df)
            
            cell_density_list.append(cell_density)
        # new csv file #
        new_df = pd.DataFrame({"structure_acronym":[],"structure_name":[], "cell_density":[]})
        for i in range(len(brain_region)):
            new_df.loc[i+1,"structure_acronym"] = brain_region[i]
            new_df.loc[i+1,"structure_name"] = brain_region_name[i]
            new_df.loc[i+1,"cell_density"] = cell_density_list[i]
        
        new_df.to_csv(savepath + "\\" + group + "\\"+ cfos_csv, index = False)   
