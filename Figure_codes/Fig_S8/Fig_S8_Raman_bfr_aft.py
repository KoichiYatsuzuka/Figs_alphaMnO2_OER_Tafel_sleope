#%%
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
import numpy as np
import pandas as pd
from collections import namedtuple

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import F, R, T
from Common_for_figs import Point
Common_for_figs.set_common_matlptlib_rcParameter()
os.chdir(Path(__file__).resolve().parent.__str__())

def get_relative_pos(value_in_datascale, axis_min, axis_max):
    return (value_in_datascale-axis_min)/(axis_max-axis_min)

"""CV 60回後のラマンスペクトル
"""

def Raman_before_after_CV():
    dataframe = pd.read_csv("20230928_2-54_BfrAftCV_finely-calibrated/comparison.txt", sep="\t", skiprows=29)
    #dataframe_after_CV = pd.read_csv("20230222_2-6_001_RamanAfterCV.txt")

    #スペクトル規格化
    sub_bl_before = dataframe['N_0CV'].values-min(dataframe['N_0CV'].values)
    sub_bl_after = dataframe['N_60CV'].values-min(dataframe['N_60CV'].values)

    nolmed_before_CV = sub_bl_before/max(sub_bl_before)
    nolmed_after_CV = sub_bl_after/max(sub_bl_after)

    #描画準備
    fig = plt.figure(figsize = (4,3))
    ax = fig. add_axes([0.1,0.2,0.8,0.7])
    ax.set_xlabel('Wavenumber [cm$^{-1}$]')
    ax.set_ylabel('Intensity [a.u.]')
    ax.set_xlim(100, 1000)
    ax.set_ylim(0, 2)
    ax.tick_params(labelleft=False, left=False, right=False)
    
    ax.plot(
        dataframe['wn(N_0CV)'].values,
        nolmed_before_CV+0.1,
        c="#FF3333",
    )
    ax.plot(
        dataframe['wn(N_60CV)'].values,
        nolmed_after_CV+0.8,
        c="#3333FF"
    )

    #annotations
    ax.text(
        990,
        0.35,
        "before\n60 CV scans",
        c="#FF3333",
        ha= 'right'
    )

    ax.text(
        990,
        1.05,
        "after \n60 CV scans",
        c="#3333FF",
        ha= 'right'
    )
    
    fig.savefig("Raman_before_after_CV.png", dpi=600)

    """fig = plt.figure(figsize = (4,3))
    ax = fig. add_axes([0.1,0.2,0.8,0.7])
    ax.set_xlabel('Wavenumber [cm$^{-1}$]')
    ax.set_ylabel('Intensity [a.u.]')
    ax.set_xlim(100, 1000)
    ax.set_ylim(-0.1, 0.1)
    ax.tick_params(labelleft=False)

    ax.plot(
        dataframe_before_CV['wavenumber'].values,
        nolmed_after_CV-nolmed_before_CV,
        c="k",
    )"""

    return

Raman_before_after_CV()

# %%
