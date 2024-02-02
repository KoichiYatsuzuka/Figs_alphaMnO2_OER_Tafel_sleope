#%%
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import savgol_filter

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import F, R, T
os.chdir(Path(__file__).resolve().parent.__str__())

Common_for_figs.set_common_matlptlib_rcParameter()
"""印加電位によるUV-visの変化（CV前）
"""
# ----------CONST values required to change----------------------
#軸範囲
X_MIN: float = 320
X_MAX: float = 800

#電位範囲
APPLIED_POTENTIALS: list =[0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]

#図を分ける範囲
SPLIT_POTENTIAL = 200 #mV
SPLIT_INDEX = APPLIED_POTENTIALS.index(SPLIT_POTENTIAL)
# -----------------end of CONST values---------------------------

#data load
data_file_raw = pd.read_csv("UV_vis_comp_Potential_BeforeCVs.csv")
wavelength = data_file_raw["波長 (nm)"]
spectra: list=[]

data_file_smoothed = pd.read_csv("UV_vis_comp_Potential_BeforeCVs_smoothed.csv")
smoothed_spectra: list=[]

for potential in APPLIED_POTENTIALS: 
    spectra.append(data_file_raw[str(potential) + " mV"])
    smoothed_spectra.append(data_file_smoothed[str(potential) + " mV"])

#---------------------------------------------------------------------------
#----------------------(A) 0V~0.2V------------------------------------------
#---------------------------------------------------------------------------

# 描画範囲準備
fig = plt.figure(figsize = (4,3))
ax = fig. add_axes([0.23,0.2,0.72,0.7])
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel(r'$\Delta$Absorbance')
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-0.015, 0.05)

#plot
for i in range(SPLIT_INDEX+1):#0 mV~ 200 mVまで
    #color_rgb_raw = (i/float(SPLIT_INDEX+1)*0.8+0.2, 0.2, 0.2)
    color_rgb_raw = (i/float(SPLIT_INDEX)*0.7+0.2, 0.2, 0.2)
    color_rgb_smoothed = (i/float(SPLIT_INDEX)*0.7+0.2, 0.3, 0.3)
    
    ax.plot(wavelength, spectra[i]-spectra[0], c=color_rgb_raw, alpha=0.3, lw=0.5)
    ax.plot(wavelength, smoothed_spectra[i]-smoothed_spectra[0], c=color_rgb_smoothed, alpha=1.0)

# annotations
ax.annotate(
    text="",
    xy=(620,0.04),
    xytext=(620, 0.02),
    xycoords="data",
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3"
    )
)

ax.text(x=600,y=0.017,s="0.2 V")
ax.text(x=600,y=0.041,s="0.4 V")
ax.text(x=200, y=0.05, s="(A)")

plt.savefig("UVvis_potentialDependency_0_200.png", dpi=600)

#---------------------------------------------------------------------------
#----------------------(B) 0.2V~0.8V----------------------------------------
#---------------------------------------------------------------------------

# 描画範囲準備
fig = plt.figure(figsize = (4,3))
ax = fig. add_axes([0.23,0.2,0.72,0.7])
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel(r'$\Delta$Absorbance')
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-0.02, 0.05)

#plot
for i in range(SPLIT_INDEX, len(spectra)):#200 mV~ 800 mVまで
    tmp = (i-SPLIT_INDEX-1)/float(len(spectra)-SPLIT_INDEX-1)
    color_rgb_raw = (1-(i-SPLIT_INDEX)/float(len(spectra)-SPLIT_INDEX), 0.2, 0.2+(i-SPLIT_INDEX)/float(len(spectra)-SPLIT_INDEX)*0.7)
    color_rgb_smoothed = (1-(i-SPLIT_INDEX)/float(len(spectra)-SPLIT_INDEX), 0.3, 0.3+(i-SPLIT_INDEX)/float(len(spectra)-SPLIT_INDEX)*0.6)
    
    ax.plot(wavelength, spectra[i]-spectra[4], c=color_rgb_raw, alpha=0.3, lw=0.5)
    ax.plot(wavelength, smoothed_spectra[i]-smoothed_spectra[4], c=color_rgb_smoothed, alpha=1.0)

# annotations
ax.annotate(
    text="",
    xy=(670,0.04),
    xytext=(670, 0.025),
    xycoords="data",
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3"
    )
)

ax.text(x=650,y=0.02,s="0.4 V")
ax.text(x=650,y=0.042,s="1.0 V")
ax.text(x=200, y=0.05, s="(B)")
plt.savefig("UVvis_potentialDependency_300_800.png", dpi=600)
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""

#%%
"""CV掃引前のUV-vis (100 mV間隔の差スペクトル)
"""
#-----------CONSTs-------------------------
#カラム名リスト
COLUMNS: list =[
    "100-0",
    "200-100",
    "300-200",
    "400-300",
    "500-400",
    "600-500",
    "700-600",
    "800-700"
    ]
#-----------------------------------------------------------
# 元データ読み込み
data_file = pd.read_csv("UV-vis_comp_Potential_BeforeCVs.csv")
wavelength = data_file["波長 (nm)"]
spectra: list[np.ndarray]=[]
for column in COLUMNS:
    spectra.append(data_file[column].values)
# なんとかフィルターかけた後のデータ読み込み
data_file2 = pd.read_csv("UV-vis_comp_Potential_BeforeCVs_smoothed.csv")
smoothed_spectra: list[np.ndarray] =[]
for column in COLUMNS:
    smoothed_spectra.append(data_file2[column])

# 描画範囲設定
fig = plt.figure(figsize = (4,7))
ax = fig. add_axes([0.12,0.1,0.8,0.85])
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel(r'$\Delta$absorbance')
ax.set_xlim(320, 800)
ax.set_ylim(-0.185, 0.03)
ax.tick_params(labelleft=False, left=False, right=False)

#plot
y_offset: float = -0.025
for index,spectrum in enumerate(spectra):
    color_rgb = (1-index/float(len(spectra))*0.7,0.3,index/float(len(spectra))*0.7+0.3)
    ax.plot(wavelength, spectrum+y_offset*index, c=color_rgb, alpha=0.3, lw=0.5)
    
for index,spectrum in enumerate(smoothed_spectra):
    color_rgb = (1-index/float(len(spectra))*0.8,0.2,index/float(len(spectra))*0.8+0.2)
    ax.plot(wavelength, spectrum+y_offset*index, c=color_rgb, alpha=1.0)



# annotations
## 各々の差スペクトルに電位差を表示
for index in range(len(spectra)):
    higher_potential:float = 0.3 + index*0.1
    lower_potential:float = 0.2 + index*0.1
    ax.text(
        x=620, y=0.008+y_offset*index, 
        s="{:.1f} V - {:.1f} V".format(higher_potential, lower_potential)
        )
# ピークトップの線
ax.axvline(485, c="k", linestyle=":", lw=1.0)
ax.axvline(520, c="k", linestyle=":", lw=1.0)

## delta-abs = 0の補助線
for index,spectrum in enumerate(spectra):
    ax.axhline(y=0+y_offset*index, linestyle="--", lw=1.0, c="k")

## 吸光度の変位の大きさを示す両矢印
ax.annotate(
    text="",
    xy=(350,0.027),
    xytext=(350, 0.017),
    xycoords="data",
    arrowprops=dict(\
        arrowstyle = ArrowStyle("|-|", widthA=0.3, widthB=0.3),
        connectionstyle="arc3"
    )
)
ax.text(360, 0.02, s=r"$\Delta$=0.01")



plt.savefig("UVvis_delta_absorbance_each_100mV", dpi=600)
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
