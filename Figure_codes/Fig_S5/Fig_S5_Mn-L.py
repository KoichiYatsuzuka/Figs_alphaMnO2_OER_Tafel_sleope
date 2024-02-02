"""
XAFS
材料の同定、電解前後の比較
"""
#%%
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import seaborn as sns
from copy import deepcopy as copy
import pandas as pd

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import color_map_RGB, color_map_RtoB
from Common_for_figs import F, R, T
from Common_for_figs import SSCE_TO_SHE
os.chdir(Path(__file__).resolve().parent.__str__())

Common_for_figs.set_common_matlptlib_rcParameter()

def XAFS_BL17SU_file_read(file_name: str, header_line_num = 12)->pd.DataFrame:
    col_names = [
        "pulse",
        "Energy",
        "Absorption",
        "ID",
        "step"
    ]
    data_frame = pd.read_csv(file_name, sep="\t", header=header_line_num, names=col_names)
    return data_frame


def Soft_XAFS():

    file_list_0CV =[
        "MnO2_CVs_Mn_L_r1001.DAT",
        "MnO2_CVs_Mn_L_r2001.DAT",
        "MnO2_CVs_Mn_L_r3001.DAT",
        "MnO2_CVs001.DAT"
    ]

    file_list_60CV =[
        "MnO2_CVs_Mn_L_r1002.DAT",
        "MnO2_CVs_Mn_L_r2002.DAT",
        "MnO2_CVs_Mn_L_r3002.DAT",
        "MnO2_CVs004.DAT"
    ]

    def read_from_file_list(file_list: list):
        tmp_df_list = []
        for file_name in file_list:
            tmp_df = XAFS_BL17SU_file_read(file_name)
            tmp_df_list.append(copy(tmp_df))
        return copy(tmp_df_list)

    """df_list_0CV = read_from_file_list(file_list_0CV)
    df_list_60CV = read_from_file_list(file_list_60CV)"""

    df_0CV = XAFS_BL17SU_file_read("MnO2_CVs_Mn_L_r1001.DAT")
    df_60CV = XAFS_BL17SU_file_read("MnO2_CVs_Mn_L_r1002.DAT")

    ## reference samples
    df_MnO2 = XAFS_BL17SU_file_read("MnO2_Mn_3p001.DAT")
    df_Mn2O3 = XAFS_BL17SU_file_read("Mn2O3_Mn_3pMn2O3_MnL_3002.DAT")

    def calc_avetrage_deviation(df_list: list):
        array_absorption_list = np.zeros((len(df_list), df_list[0]['Absorption'].values.size))
        array_norm_absorption_list = np.zeros((len(df_list), df_list[0]['Absorption'].values.size))

        for index, df in enumerate(df_list):
            spectrum = copy(df['Absorption'].values)
            array_absorption_list[index]=spectrum
            array_norm_absorption_list[index] = (spectrum-np.min(spectrum))/np.max(spectrum-np.min(spectrum))
        
        df_avr = pd.DataFrame()
        df_avr["Energy"] = df_list[0]['Energy']
        df_avr["Absorption"] = np.mean(array_absorption_list, axis=0)
        df_avr["Std. deviation"] = np.std(array_absorption_list, axis=0)
        df_avr["Norm. absorption"] = np.mean(array_norm_absorption_list, axis=0)
        df_avr["Norm. std. deviation"] = np.std(array_norm_absorption_list, axis=0)

        return df_avr

    """df_avr_0CV = calc_avetrage_deviation(df_list_0CV)
    df_avr_60CV = calc_avetrage_deviation(df_list_60CV)"""
    
    """#描画（同定用の図）
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.1,0.2,0.8,0.7])
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Intensity [a.u.]")
 
    ax.set_xticks(np.linspace(630, 660, 4))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlim(632, 664)
    ax.set_ylim(-0.01, 1.4)
    plt.tick_params(labelleft=False, left=False, right=False)

    ax.plot(
        df_avr_0CV["Energy"].values, 
        df_avr_0CV["Norm. absorption"].values,
        lw = 1.0,
        c="k"
        )
    
    #ax.vlines(642, ymin=0, ymax=1, linestyles="dotted")
    
    # annotations

    ##矢印の書式
    arrow_type = dict(
        arrowstyle = "->",
        color="k"
    )
    ax.text(
        -0.1, 1,
        "(C)",
        transform = ax.transAxes
    )

    ## 帰属
    ax.annotate(
        "Mn 2p$_{3/2}$ → Mn 3d(t$_{2g}$)",
        (641, 0.95),
        (633, 1.25),
        arrowprops = arrow_type
    )
    ax.annotate(
        "Mn 2p$_{3/2}$ → Mn 3d(e$_g$)",
        (644, 1.01),
        (645, 1.1),
        arrowprops = arrow_type
    )
    ax.annotate(
        "Mn 2p$_{1/2}$ → Mn 3d",
        (644, 1.01),
        (647, 0.8),
        #arrowprops = arrow_type
    )

    plt.savefig("XAFS_soft_characterization", dpi=600)"""

    #描画（電解比較用）
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.1,0.2,0.8,0.7])
    ax.set_xlabel("Photon energy [eV]")
    ax.set_ylabel("Normalized intensity [a.u.]")
 
    ax.set_xticks(np.linspace(630, 660, 4))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlim(635, 660)
    ax.set_ylim(-0.1, 1.5)

    plt.tick_params(labelleft=False, left=False, right=False)

    def normalize_spectrum(spectrum: np.ndarray):
        return (spectrum-np.min(spectrum))/np.max(spectrum-np.min(spectrum))

    """ ax.fill_between(
        df_avr_0CV["Energy"].values,
        df_avr_0CV["Norm. absorption"].values + df_avr_0CV["Norm. std. deviation"].values,
        df_avr_0CV["Norm. absorption"].values - df_avr_0CV["Norm. std. deviation"].values,
        facecolor = "#8888FF",
        alpha = 0.5
        )  """
    ax.plot(
        df_0CV["Energy"].values, 
        normalize_spectrum(df_0CV["Absorption"]),
        lw = 1.0,
        c="#FF3333",
        label = r"Before 60 CV scans",
        linestyle="--"
        )
    
    """ ax.fill_between(
        df_avr_60CV["Energy"].values,
        df_avr_60CV["Norm. absorption"].values + df_avr_60CV["Norm. std. deviation"].values+0.4,
        df_avr_60CV["Norm. absorption"].values - df_avr_60CV["Norm. std. deviation"].values+0.4,
        facecolor = "#FF8888",
        alpha = 0.5
        ) """
    ax.plot(
        df_60CV["Energy"].values, 
        normalize_spectrum(df_60CV["Absorption"]),
        lw = 1.0,
        c="#3333FF",
        label = r"After 60 CV scans",
        linestyle = "--"
        )
    """ ax.plot(
        df_avr_60CV["Energy"].values, 
        (df_avr_60CV["Norm. absorption"].values-df_avr_0CV["Norm. absorption"].values)*50,
    ) """
    """ax.text(
        632.5,0.05,
        "Before \n60 CV\nscans",
        c="#FF3333"
    )
    ax.text(
        632.5,0.55,
        "After \n60 CV\nscans",
        c="#3333FF"
    )"""
    
    ## reference sampleのプロット
    
    """ax.plot(
        df_MnO2["Energy"].values,
        normalize_spectrum(df_MnO2["Absorption"].values),
        lw = 1.0,
        c="#338833",
        linestyle= "--",
        label = r"$\delta$-MnO$_2$ (reference)"
        )"""


    ax.plot(
        df_Mn2O3["Energy"].values,
        normalize_spectrum(df_Mn2O3["Absorption"].values),
        lw=1.0,
        c="#888833",
        linestyle="--",
        label = "Mn$_2$O$_3$"
        )

    ax.text(
        640,1.05,
        "Mn L$\mathrm{_{III}}$",
        c="k"
    )
    ax.text(
        652,0.85,
        "Mn L$\mathrm{_{II}}$",
        c="k"
    )
    ax.legend()
    plt.savefig("XAFS_soft_before_after.png", dpi=600)

    """ fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.1,0.2,0.8,0.7])
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("absorption [a.u.]")
 
    ax.set_xticks(np.linspace(630, 660, 4))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlim(632, 664)

    c_list=["k", "r", "g", "b"]
    for i, df in enumerate(df_list_0CV):

        ax.plot(
            df_avr_0CV["Energy"].values, 
            df["Absorption"].values,
            lw = 1.0,
            c=c_list[i]
            )
 """
Soft_XAFS()
# %%
