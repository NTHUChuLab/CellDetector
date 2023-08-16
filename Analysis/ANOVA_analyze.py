import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from natsort import natsorted


def get_path_and_option():
    brain_path = input("INSERT THE PATH OF SUMMARY OUTPUT :\n")
    brain_list_path = input("INSERT THE PATH OF BRAIN REGION LIST :\n")
    behavior_path = input("INSERT THE PATH OF BEHAVIOR RESULT :\n")
    spearsman_or_linear_regression = input("spearsman or linear regression(1 means spearsman, other means linear regression) :\n") #要做線性回歸還是spearsman
    try:
        if int(spearsman_or_linear_regression) == 1:
            spearsman_or_linear_regression = True
            dom_and_sub_together_or_not = ''
        else:
            spearsman_or_linear_regression = False
    except:
        spearsman_or_linear_regression = False
    
    if not spearsman_or_linear_regression:
        dom_and_sub_together_or_not = input("DOM AND SUB ANALYZE TOGETHER OR NOT(1 means together, other means seperate) :\n")
        
        try:
            if int(dom_and_sub_together_or_not) == 1:
                dom_and_sub_together_or_not = True
            else:
                dom_and_sub_together_or_not = False
        except:
            dom_and_sub_together_or_not = False
    
    
    result_path = brain_path + '/result'
    
    os.makedirs(result_path, exist_ok = True)
    
    return brain_path, brain_list_path, behavior_path, result_path, dom_and_sub_together_or_not, spearsman_or_linear_regression


def get_structure_list(brain_list_path):
    structure_list = []
    
    with open(brain_list_path) as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            structure_list.append(str(row['name']))
    
    return structure_list
   

def from_csv_get_cell_density(path, structure):
    with open(path) as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            if row['structure_name'] == structure:
                
                cell_number = float(row['total_cells'])
                brain_region_density = float(row['total_volume_mm3'])
                if brain_region_density == 0:
                    cell_density = 0
                    break
                
                cell_density = (cell_number/brain_region_density)
                #cell_density = float(row['avg_intensity']) # intensity
                break
    try:
        cell_density = cell_density+1-1
    except:
        cell_density = 0
    
    return cell_density



def fill_cfos(brain_path, group, structure, df, D_vs_Sdf, mouse_number):
    files = natsorted(os.listdir(brain_path + '/' + group))
    for dfnumber in range(mouse_number):
        cell_density = from_csv_get_cell_density(brain_path + '/' + group + '/' + files[dfnumber], structure)
        df.loc[dfnumber, 'cfos'] = cell_density
        D_vs_Sdf.loc[dfnumber, group] = cell_density
    
    return df, D_vs_Sdf


def fill_behavior(mouse_number, df, behavior_path, behavior_group):
    behavior_df = pd.read_excel(behavior_path)
    for dfnumber in range(mouse_number):
        behavior_result = behavior_df.loc[dfnumber, behavior_group]
        df.loc[dfnumber, 'behavior'] = behavior_result
    
    return df    


def create_dataframe(brain_path, group, structure, behavior_path, behavior_group, savepath, group_number, D_vs_Sdf, mouse_number):
    df = pd.DataFrame({"cfos":[], "behavior":[]})
    if group_number == 0:
        D_vs_Sdf = pd.DataFrame({"dominant":[], "subordinate":[]})
    
    df, D_vs_Sdf = fill_cfos(brain_path, group, structure, df, D_vs_Sdf, mouse_number)
    df = fill_behavior(mouse_number, df, behavior_path, behavior_group)
    df.to_excel(savepath + behavior_group + '.xlsx', index = False) #my
    if group_number == 1:
        D_vs_Sdf.to_excel(savepath + 'D_vs_S.xlsx', index = False)
    
    return df, D_vs_Sdf


def ctrl_dom_sub_anova(ctrl, dom, sub, savepath):
    F, anova_p = stats.f_oneway(ctrl, dom, sub)
    if anova_p <= 0.05:
        df = pd.DataFrame({'score': np.hstack((ctrl, dom, sub)),
                   'group': np.repeat(['ctrl', 'dom', 'sub'], repeats=10)})
        tukey = pairwise_tukeyhsd(endog=df['score'],
                          groups=df['group'],
                          alpha=0.05)
        path = savepath + 'tukey.txt'
        f = open(path, 'w')
        f.write(str(tukey))
        f.close()
    return anova_p


def spearsman(df):
    a = np.array(df['behavior'])
    b = np.array(df['cfos'])
    correlation, spearsman_pvalue = stats.spearmanr(a, b)
    
    return correlation, spearsman_pvalue

def pearson(df):
    a = np.array(df['behavior'])
    b = np.array(df['cfos'])
    pearson_R, pearson_P = stats.pearsonr(a, b)
    
    return pearson_R, pearson_P


def fill_anova(statistics_path, table_number, brain_path, mouse_number, savepath, D_vs_Sdf, structure):
    table_df = pd.read_excel(statistics_path)
    files = natsorted(os.listdir(brain_path + '/control'))
    for dfnumber in range(mouse_number):
        cell_density = from_csv_get_cell_density(brain_path + '/control/' + files[dfnumber], structure)
        D_vs_Sdf.loc[dfnumber, 'control'] = cell_density
    D_vs_Sdf.to_excel(savepath + 'D_vs_S.xlsx', index = False)
    table_df.loc[table_number, 'ANOVA'] = ctrl_dom_sub_anova(D_vs_Sdf['control'], D_vs_Sdf['dominant'], D_vs_Sdf['subordinate'], savepath)
    table_df.to_excel(statistics_path, index = False)


def fill_spearsman(correlation, spearsman_pvalue, column_number, table_number, structure, result_path):
    if table_number == 0:
        table_df = pd.DataFrame({"brain_region":[], "spearman_aggression_10_R":[], "spearman_aggression_10_P":[], "spearman_grooming_10_R":[], "spearman_grooming_10_P":[], "spearman_aggression_20_R":[], "spearman_aggression_20_P":[], "spearman_grooming_20_R":[], "spearman_grooming_20_P":[], "total_social_time_spearman_R_dom":[], "total_social_time_spearman_P_dom":[], "total_social_time_spearman_R_sub":[], "total_social_time_spearman_P_sub":[], "total_social_time_spearman_R_dom+sub":[], "total_social_time_spearman_P_dom+sub":[]})
    else:
        table_df = pd.read_excel(result_path + '/statistics.xlsx')
    
    table_df.loc[table_number, 'brain_region'] = structure
    table_df.iloc[table_number, column_number] = correlation
    table_df.iloc[table_number, column_number+1] = spearsman_pvalue
    
    table_df.to_excel(result_path + '/statistics.xlsx', index = False)
    
    
def fill_pearson(pearson_R, pearson_P, table_number, structure, result_path, group):
    table_df = pd.read_excel(result_path + '/statistics.xlsx')
    table_df.loc[table_number, 'brain_region'] = structure
    
    table_df.loc[table_number, 'total_social_time_pearson_R_'+str(group)] =  pearson_R
    table_df.loc[table_number, 'total_social_time_pearson_P_'+str(group)] =  pearson_P
    
    table_df.to_excel(result_path + '/statistics.xlsx', index = False)



def D_vs_S_t_test(D_vs_Sdf): # Dom vs Sub (cfos only, no behavior)
    statistic, pvalue = stats.ttest_rel(D_vs_Sdf.loc[:, 'dominant'], D_vs_Sdf.loc[:, 'subordinate'])
    
    return pvalue

def fill_t_test(result_path, structure, D_vs_Sdf, table_number):
    table_df = pd.read_excel(result_path + '/statistics.xlsx')
    table_df.loc[table_number, 'brain_region'] = structure
    table_df.loc[table_number, 'D_vs_S_P'] = D_vs_S_t_test(D_vs_Sdf)
    
    table_df.to_excel(result_path + '/statistics.xlsx', index = False)
    
    
def create_table(coef, p_value, result_path, group_number, table_number, structure, D_vs_Sdf):
    if table_number == 0:
        table_df = pd.DataFrame({"brain_region":[], "aggression_R":[], "aggression_P":[], "grooming_R":[], "grooming_P":[], "D_vs_S_P":[]})
        table_df.loc[table_number, 'brain_region'] = structure
        table_df.loc[table_number, 'aggression_R'] = coef
        table_df.loc[table_number, 'aggression_P'] = p_value
    else:
        table_df = pd.read_excel(result_path + '/statistics.xlsx')
    
    if group_number == 0:
        table_df.loc[table_number, 'brain_region'] = structure
        table_df.loc[table_number, 'aggression_R'] = coef
        table_df.loc[table_number, 'aggression_P'] = p_value
    else:
        table_df.loc[table_number, 'grooming_R'] = coef
        table_df.loc[table_number, 'grooming_P'] = p_value
        table_df.loc[table_number, 'D_vs_S_P'] = D_vs_S_t_test(D_vs_Sdf)
    
    table_df.to_excel(result_path + '/statistics.xlsx', index = False)


def create_figure(structure, df, savepath, result_path, group_number, table_number, D_vs_Sdf, option):
    plt.xlabel('behavior result')
    plt.ylabel('cfos density')
    plt.title(structure)
    
    X = np.array(df['behavior'])
    Y = np.array(df['cfos'])
    
    plt.scatter(X, Y)
    
    X_modified = X[:, np.newaxis]
    
    straight_line = LinearRegression()
    straight_line.fit(X_modified, Y)
    
    xfit = np.linspace(0, max(X), 10)
    yfit = straight_line.predict(xfit[:, np.newaxis])
    
    plt.plot(xfit, yfit)
    
    '''
    if sum(Y) < 8:
        coef = 0
        p_value = 0
    else:
        coef = straight_line.score(X_modified, Y)
        p_value = f_regression(X_modified,Y)[1]
    
    create_table(coef, p_value, result_path, group_number, table_number, structure, D_vs_Sdf)
    '''
    plt.savefig(savepath + '.png')
    
    plt.cla()


def concatenate_grooming(data_path, mouse_number):
    sub_data = pd.read_excel(data_path + 'subordinate_allogrooming.xlsx')
    dom_data = pd.read_excel(data_path + 'dominant__allogrooming.xlsx')
    data_df = pd.DataFrame({"cfos":[], "behavior":[]})
    for dfnumber in range(mouse_number):
        sub_cfos = sub_data.loc[dfnumber, 'cfos']
        data_df.loc[dfnumber, 'cfos'] = sub_cfos
        sub_behavior = sub_data.loc[dfnumber, 'behavior']
        data_df.loc[dfnumber, 'behavior'] = sub_behavior
    
    for dfnumber in range(mouse_number, 2 * mouse_number):
        dom_cfos = dom_data.loc[dfnumber - mouse_number, 'cfos']
        data_df.loc[dfnumber, 'cfos'] = dom_cfos
        dom_behavior = dom_data.loc[dfnumber - mouse_number, 'behavior']
        data_df.loc[dfnumber, 'behavior'] = dom_behavior
    
    data_df.to_excel(data_path + 'grooming.xlsx', index = False)
    return data_df


def concatenate_aggression(data_path, mouse_number):
    sub_data = pd.read_excel(data_path + 'subordinate_aggression.xlsx')
    dom_data = pd.read_excel(data_path + 'dominant_aggression.xlsx')
    data_df = pd.DataFrame({"cfos":[], "behavior":[]})
    for dfnumber in range(mouse_number):
        sub_cfos = sub_data.loc[dfnumber, 'cfos']
        data_df.loc[dfnumber, 'cfos'] = sub_cfos
        sub_behavior = sub_data.loc[dfnumber, 'behavior']
        data_df.loc[dfnumber, 'behavior'] = sub_behavior
    
    for dfnumber in range(mouse_number, 2 * mouse_number):
        dom_cfos = dom_data.loc[dfnumber - mouse_number, 'cfos']
        data_df.loc[dfnumber, 'cfos'] = dom_cfos
        dom_behavior = dom_data.loc[dfnumber - mouse_number, 'behavior']
        data_df.loc[dfnumber, 'behavior'] = dom_behavior
    
    data_df.to_excel(data_path + 'aggression.xlsx', index = False)
    return data_df

def concatenate_total_social_time(data_path, mouse_number):
    sub_data = pd.read_excel(data_path + 'sub_total_social_time.xlsx')
    dom_data = pd.read_excel(data_path + 'dom_total_social_time.xlsx')
    data_df = pd.DataFrame({"cfos":[], "behavior":[]})

    for dfnumber in range(mouse_number):
        dom_cfos = dom_data.loc[dfnumber, 'cfos']
        data_df.loc[dfnumber, 'cfos'] = dom_cfos
        dom_behavior = dom_data.loc[dfnumber, 'behavior']
        data_df.loc[dfnumber, 'behavior'] = dom_behavior
    
    for dfnumber in range(mouse_number, 2 * mouse_number):
        sub_cfos = sub_data.loc[dfnumber - mouse_number, 'cfos']
        data_df.loc[dfnumber, 'cfos'] = sub_cfos
        sub_behavior = sub_data.loc[dfnumber - mouse_number, 'behavior']
        data_df.loc[dfnumber, 'behavior'] = sub_behavior
    
    data_df.to_excel(data_path + 'total_social_time.xlsx', index = False)
    return data_df


def create_dataframe2(brain_path, group, mouse_number, structure, behavior_path, behavior_group, savepath):
    df = pd.DataFrame({"cfos":[], "behavior":[]})
    files = natsorted(os.listdir(brain_path + '/' + group))
    for dfnumber in range(mouse_number):
        cell_density = from_csv_get_cell_density(brain_path + '/' + group + '/' + files[dfnumber], structure)
        df.loc[dfnumber, 'cfos'] = cell_density
    behavior_df = pd.read_excel(behavior_path)
    for dfnumber in range(mouse_number):
        behavior_result = behavior_df.loc[dfnumber, behavior_group]
        df.loc[dfnumber, 'behavior'] = behavior_result
    
    df.to_excel(savepath + behavior_group + '.xlsx', index = False)
    return df


def create_brain_region_result(result_path, structure, brain_path, group_list, behavior_path, behavior_groups, structure_modified, table_number, option1, mouse_number, behavior_groups2, option2, total_social_time):
    os.makedirs(result_path + '/' + structure_modified, exist_ok = True)
    os.makedirs(result_path + '/' + structure_modified + '/excel', exist_ok = True)
    os.makedirs(result_path + '/' + structure_modified + '/figure', exist_ok = True)
    
    if not option2:
        D_vs_Sdf = []
        for group_number in range(2):
            df, D_vs_Sdf = create_dataframe(brain_path, group_list[group_number], structure, behavior_path, behavior_groups[group_number], result_path + '/' + structure_modified + '/excel/', group_number, D_vs_Sdf, mouse_number)
        if option1:
            for group_number in range(2):
                create_dataframe2(brain_path, group_list[group_number], mouse_number, structure, behavior_path, behavior_groups2[group_number], result_path + '/' + structure_modified + '/excel/')
            grooming_data_df = concatenate_grooming(result_path + '/' + structure_modified + '/excel/', mouse_number)
            aggression_data_df = concatenate_aggression(result_path + '/' + structure_modified + '/excel/', mouse_number)
            create_figure(structure, aggression_data_df, result_path + '/' + structure_modified + '/figure/aggression', result_path, 0, table_number, D_vs_Sdf, option1)
            create_figure(structure, grooming_data_df, result_path + '/' + structure_modified + '/figure/grooming', result_path, 1, table_number, D_vs_Sdf, option1)
            fill_anova(result_path + '/statistics.xlsx', table_number, brain_path, mouse_number, result_path + '/' + structure_modified + '/excel/', D_vs_Sdf, structure)
        else:
            for group_number in range(2):
                create_figure(structure, df, result_path + '/' + structure_modified + '/figure/' + group_list[group_number], result_path, group_number, table_number, D_vs_Sdf, option1)
    else:
        D_vs_Sdf = []
        for group_number in range(2):
            df, D_vs_Sdf = create_dataframe(brain_path, group_list[group_number], structure, behavior_path, behavior_groups[group_number], result_path + '/' + structure_modified + '/excel/', group_number, D_vs_Sdf, mouse_number)
                       
        fill_anova(result_path + '/statistics.xlsx', table_number, brain_path, mouse_number, result_path + '/' + structure_modified + '/excel/', D_vs_Sdf, structure)



def main():
    group_list = ['dominant', 'subordinate']
    behavior_groups = ['dominant_aggression', 'subordinate_allogrooming']
    behavior_groups2 = ['dominant__allogrooming', 'subordinate_aggression']
    total_social_time = ['dom_total_social_time', 'sub_total_social_time']
    
    path_and_option = get_path_and_option()
    structrures = get_structure_list(path_and_option[1])
    
    files = natsorted(os.listdir(path_and_option[0] + '/dominant'))
    mouse_number = len(files)
    
    table_number = 0
    
    for structure in tqdm(structrures):
        
        if '/' in structure:
            structure_modified = structure.replace("/", "")
        else:
            structure_modified = structure
        
        create_brain_region_result(path_and_option[3], structure, path_and_option[0], group_list, path_and_option[2], behavior_groups, structure_modified, table_number, path_and_option[4], mouse_number, behavior_groups2, path_and_option[5], total_social_time)
        
        table_number += 1

main()