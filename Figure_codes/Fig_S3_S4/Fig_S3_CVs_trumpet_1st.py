"""
実測ボルタモグラム
数値シミュレーションはSimu, トランペット解析部分は
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec
from collections import namedtuple

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import color_map_RGB, color_map_RtoB
from Common_for_figs import F, R, T
from Common_for_figs import SSCE_TO_SHE
Common_for_figs.set_common_matlptlib_rcParameter()
os.chdir(Path(__file__).resolve().parent.__str__())

# CVs in several scanrate
def trumpet_CV():
    """
    Trumpet用に測定したCVを重ね書きする。
    ファイル名のベース部分を与えることで、該当する複数のファイルを自動で読み込む
    y軸は掃引速度で規格化する。（mA/(mV/s)になる）
    """
    #どのファイルの番号がどの掃引速度に対応するかのtuple
    FILE_NUM_AND_SCANRATE_BEFORE: list =[
        #(file_num, scanrate mV/s)
        ('04', 1),('05', 5),('06', 10),('07', 20),('08', 40),
        ('09', 60),('10', 80),('11', 100),('12', 200),('13', 400),
        ('14', 600),('15', 800),('16', 1000)
    ]
    FILE_NUM_AND_SCANRATE_AFTER: list =[
        #(file_num, scanrate mV/s)
        ('02', 1),('03', 5),('04', 10),('05', 20),('06', 40),
        ('07', 60),('08', 80),('09', 100),('10', 200),('11', 400),
        ('12', 600),('13', 800),('14', 1000)
    ]

    file_basename_before = '20221101_1-115_002_TrumpetPlot_ParaChange_{}_CV_C01_1.csv'
    file_basename_after = '20221101_1-115_003_TrumpetPlot_ParaChange_from10_every3_{}_CV_C01_16.csv'

    #軸範囲
    X_MIN: float = -0.01
    X_MAX: float = 0.51
    Y_MIN: float = -0.12
    Y_MAX: float = 0.12
    # -----------------end of CONST values---------------------------

    def draw_trumpet_CVs(
            file_basename: str, 
            filenum_scanrate_map: list, 
            figure_name: str
        ):
        """
            file_basenameに該当するファイルをすべて読み込み、
            filenum_scanrate_mapでファイル番号とscanrateを対応させて、
            規格化されたボルタモグラムを表示する
        引数:
            file_basename: 読み込むファイル名の一般名
            filenum_scanrate_map: 連続測定の測定番号の対応マッピング
            figure_name: 保存する画像名
        """
        index: int = 0 # just for the color of the plots
        # set canvas of plot
        fig = plt.figure(figsize = (4,3))
        ax = fig. add_axes([0.23,0.2,0.72,0.7])
        ax.set_xlabel('$E$ [V vs. SHE]')
        ax.set_ylabel(r'$j$ [mA/cm$^2$]/$\nu$ [mV/s]')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        for file_num, scan_rate in filenum_scanrate_map:
            # load and process csv data
            file_name = file_basename.format(file_num)
            data_file = pd.read_csv(file_name)
            potential=data_file['Ewe/V']+SSCE_TO_SHE

            # conversion from mA to mA/(mV/s)
            normalized_current: list = []
            for current_str in data_file['<I>/mA']:
                current_fl:float = float(current_str) 
                normalized_current.append(current_fl/scan_rate/2) #mA->mA/cm2
            
            #plot 
            color_rgb = color_map_RtoB(index/(len(filenum_scanrate_map)-1))

            ax.plot(potential, normalized_current, color=color_rgb, lw=1)
            
            index+=1
        plt.savefig(figure_name, dpi=600)
    # The end of draw_trumpet_CVs()
    
    draw_trumpet_CVs(file_basename_before, FILE_NUM_AND_SCANRATE_BEFORE,"EC_scanrate_dependency_1st.png")
    draw_trumpet_CVs(file_basename_after, FILE_NUM_AND_SCANRATE_AFTER,"EC_scanrate_dependency_60th.png")
trumpet_CV()

# %%
