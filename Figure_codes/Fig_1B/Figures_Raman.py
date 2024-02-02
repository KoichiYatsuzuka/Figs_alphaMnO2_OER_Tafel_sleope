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

"""ラマン
***同定用、CV一回掃引後のラマンスペクトル
***
***
"""
def Raman_characterization():
    #ファイル読み込み
    data_frame = pd.read_csv("20230928_2-54_BfrAftCV_finely-calibrated/comparison.txt", sep="\t", skiprows=29)
    wavenumber: np.ndarray = data_frame['wn(N_0CV)'].values
    raman_intensity_raw: np.ndarray = data_frame['N_0CV'].values
    raman_intensity_relative \
        = (raman_intensity_raw - min(raman_intensity_raw))\
        / max(raman_intensity_raw - min(raman_intensity_raw))

    #振動モード(Ref. Tanja Barudvzija, J. Alloys Compd., 2017)
    VibrationAnnotation = \
        namedtuple('VibrationAnnotation', 'wavenumber x_rel_pos y_rel_pos vibration_mode')
    
    Vib = VibrationAnnotation
    vibration_annotations = [
        Vib(181, 0, 0.1, "$E_g$"),
        Vib(296, -40, 0.1, "*"),
        Vib(327, 0, -0.05, "*"),
        Vib(387, 10, 0, "$E_g$"),
        Vib(470, -30, 0.3, "*"),
        Vib(513, 0, 0, "$E_g$"),
        Vib(580, 0, 0, "$A_g$"),
        Vib(630, 50, 0, "$A_g$"),
        Vib(745, 0, 0, "$A_g$"),

    ]


    peaktop_list=[]

    for peakpos in vibration_annotations:
        # データ横軸の配列のうち、もっともピーク位置に近い横軸位置を取得
        nearest_wavenum_index = np.argmin(np.abs(
            wavenumber - peakpos.wavenumber
        ))
        #その時のピークの高さ取得
        peak_hight = raman_intensity_relative[nearest_wavenum_index]
        peaktop_list.append(Point(peakpos.wavenumber, peak_hight))

    #描画準備
    Y_MAX = 1.5
    Y_MIN = 0
    fig = plt.figure(figsize = (4,3))
    ax = fig. add_axes([0.1,0.2,0.8,0.7])
    ax.set_xlabel('Wavenumber [cm$^{-1}$]')
    ax.set_ylabel('Intensity [a.u.]')
    ax.set_xlim(100, 1000)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.tick_params(labelleft=False, left=False, right=False)


    # プロット
    ax.plot(wavenumber, raman_intensity_relative, lw=1.0, c="#000000")

    #annotations
    ax.text(20, 1.4, "(B)")

    STD_ARROW_HIEGHT = 0.2
    for i,peaktop in enumerate(peaktop_list):

        ax.annotate(
            text=vibration_annotations[i].vibration_mode+"\n"+\
                str(vibration_annotations[i].wavenumber),
            xy=(peaktop.x,peaktop.y),
            xytext=(
                peaktop.x + vibration_annotations[i].x_rel_pos,
                peaktop.y + vibration_annotations[i].y_rel_pos + STD_ARROW_HIEGHT
            ),
            xycoords="data",
            ha="center",
            arrowprops=dict(\
                arrowstyle = ArrowStyle("->", widthA=0.3, widthB=0.3),
                connectionstyle="arc3"
            )
        )



    plt.savefig("Raman_characterization_MnO2.png", dpi=600)
Raman_characterization()

# %%
