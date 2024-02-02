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

# 共通部分の定義読み込み
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
import Common_for_figs
from Common_for_figs import color_map_RGB, color_map_RtoB
from Common_for_figs import F, R, T
from Common_for_figs import SSCE_TO_SHE
Common_for_figs.set_common_matlptlib_rcParameter()
os.chdir(Path(__file__).resolve().parent.__str__())

# 
"""CV and DPV at pH 12.5
***左右軸使っている
***CVの肩、DPVのピーク位置が同じ→同じredoxに対応している->pdvが両方で見えている
"""

#関数名微妙すぎる
def CV_DPV_pH12_5():
    # ファイル読み込み

    ## タブ区切り、偶数行だけ読む
    dataframe_DPV = pd.read_csv("20230328_2-12_003_DPV+EIS_pH12.5_01_DPV_C01.txt", sep="\t")[1::2]
    ## タブ区切り、56行読み飛ばす
    dataframe_CV = pd.read_csv("20230328_2-12_003_DPV+EIS_pH12_02_CV_C01.mpt", sep="\t", skiprows=56)

    # 描画準備
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.12,0.2,0.75,0.7])
    ax.set_xlabel('$E$ [V vs. SHE]')
    ax.set_xlim(0.6, 1.05)
    
    # 左軸の設定
    ax_CV_left = ax
    ax_CV_left.set_ylabel("$j$ [mA/cm$^2$]")
    ax_CV_left.set_ylim(-0.2, 4.5)

    ax_DPV_right = ax.twinx()
    ax_DPV_right.set_ylabel("$\Delta j$ [mA/cm$^2$]")
    

    # プロット
    ax_CV_left.plot(
        dataframe_CV["Ewe/V"].values+SSCE_TO_SHE,
        dataframe_CV['<I>/mA'].values/2,
        lw=1.0,
        c="#FF3333"
    )

    ax_DPV_right.plot(
        dataframe_DPV['E step/V'].values+SSCE_TO_SHE,
        dataframe_DPV['I delta/uA'].values/2/1000, #uA/cm2 -> mA/cm2
        lw=1.0,
        c="#3333FF"
    )


    # annotations
    ax_CV_left.annotate(
        "",
        xy=(0.9, 1.0),
        xycoords="data",
        xytext=(0.85,1.0),
        arrowprops=dict(
            arrowstyle="<-",
            connectionstyle="arc3"
        )
    )

    ax_DPV_right.annotate(
        "",
        xy=(0.872, 6),
        xycoords="data",
        xytext=(0.922,6),
        arrowprops=dict(
            arrowstyle="<-",
            connectionstyle="arc3"
        )
    )

    plt.savefig("EC_CV_DPV_pH12.5.png", dpi=600)
    pass

CV_DPV_pH12_5()
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""

# %%
