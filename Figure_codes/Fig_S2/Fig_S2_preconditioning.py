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
os.chdir(Path(__file__).resolve().parent.__str__())

Common_for_figs.set_common_matlptlib_rcParameter()


"""CV本当の最初の一回目
***活性化のやつ
"""
# ファイル読み込み
data_frame = pd.read_csv("20220924_003_MultiCV_0_850mV_1-84_C_02_CV_C01_0.csv")

corrected_potential:np.ndarray = data_frame['E-IR/V'].values+0.2
recorded_potential:np.ndarray = data_frame['Ewe/V'].values+0.2
current_density:np.ndarray = data_frame['<I>/mA'].values/2

#描画準備
fig = plt.figure(figsize = (4,3))
ax = fig. add_axes([0.2,0.2,0.7,0.7])
ax.set_xlabel('$E$ [V vs. SHE]')
ax.set_ylabel('$j$ [mA/cm$^{2}$]')
ax.set_xlim(0.2, 1.05)

#プロット
ax.plot(recorded_potential, current_density, lw=0.5, c="#505050")
ax.plot(corrected_potential, current_density, lw=1.0, c="#000000")

plt.savefig("EC_preconditioning.png")

""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
# %%
