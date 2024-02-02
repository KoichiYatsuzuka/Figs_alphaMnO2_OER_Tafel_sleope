#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#%%
DATA_FILE_NAME = "UV-vis_comp_Potential_BeforeCVs.csv"
OUTPUT_FILE_NAME = "UV-vis_comp_Potential_BeforeCVs_smoothed.csv"
WINDOW_LENGTH = 31
POLYOEDER = 3
EXCEPTION_COLUMN = [
    "波長 (nm)",
    "Raw data name",
    "Mod column name"
    ]
#%%
data_file = pd.read_csv(DATA_FILE_NAME)
new_columns:dict = {}

for column_name in data_file.keys():
    
    new_columns[column_name] = []
    
    # 例外的なカラムはそのままコピーする
    if column_name in EXCEPTION_COLUMN:
        new_columns[column_name]=data_file[column_name]
        continue

    column_tmp :list[float] =[]
    for values in data_file[column_name].values:
        try:
            column_tmp.append(float(values))
        except ValueError:
            print("value is ", values)
    new_columns[column_name]=savgol_filter(column_tmp, WINDOW_LENGTH, POLYOEDER)

output_file = open(OUTPUT_FILE_NAME, 'w', encoding = "UTF-8")

column_length: int = 0
for column_name in new_columns.keys():
    output_file.write("{},".format(column_name))
    if column_length < len(new_columns[column_name]):
        column_length = len(new_columns[column_name])
output_file.write("\n")

for i in range(0, column_length):
    for column_name in new_columns.keys():
        output_file.write("{},".format(new_columns[column_name][i]))
    output_file.write("\n")
    
output_file.close()

# %%

def make_data_absorbace_vs_potential():
    """Savitsky Golay filterかけた後の、ある波長での吸光度変化を取得
    """
    potential_list: list[int] =[
        0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800
    ]
    TARGET_WAVELENGTHES = [480, 485, 490, 495, 500, 505, 510, 515, 520]

    data_file = pd.read_csv("UV-vis_comp_Potential_BeforeCVs_smoothed.csv")
    wavelength = data_file['波長 (nm)'].tolist()

    potential_dependency_each_wavelength: list[list[str]] = []
    #[[potential dependency at AA nm],
    # [potential dependency at BB nm],
    #             ...
    # [potential dependency at ZZ nm]]

    for target_wavelength in TARGET_WAVELENGTHES:
        target_index = wavelength.index(target_wavelength)

        absorbance_list:list[str]=[]

        for potential in potential_list:
            column = data_file[str(potential)+" mV"]
            absorbance = column[target_index]
            absorbance_list.append(absorbance)

        potential_dependency_each_wavelength.append(absorbance_list)

    output_file = open("UV_vis_potential_dependency.csv", 'w', encoding="UTF-8")

    #最初のカラム名の部分
    #電位軸と追跡する吸光度
    output_file.write('potential [mV],')
    for target_wavelength in TARGET_WAVELENGTHES:
        output_file.write(str(target_wavelength)+",")
    output_file.write("\n")

    # データ部分
    for index in range(0,len(potential_list)):
        output_file.write(str(potential_list[index])+",")
        for k in range(0,len(TARGET_WAVELENGTHES)):
            output_file.write(str(potential_dependency_each_wavelength[k][index])+",")
        output_file.write("\n")
            
    output_file.close()
make_data_absorbace_vs_potential()
# %%
