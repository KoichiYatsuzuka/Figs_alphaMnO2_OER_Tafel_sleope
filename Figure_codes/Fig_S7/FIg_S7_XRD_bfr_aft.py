"""
粉末XRD

"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from collections import namedtuple

# 共通部分の定義読み込み
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
import Common_for_figs as Common
from Common_for_figs import plot_PDF
os.chdir(Path(__file__).resolve().parent.__str__())

Common.set_common_matlptlib_rcParameter()


""" XRD_before_after_CV.png
*** CV60サイクル後のXRD
"""
def PXRD_after_CVs():
	#XRDデータ読み込み
	dataframe_before_CV = pd.read_csv("20230112_2-4_006_alpha-MnO2_CV-sweeped_superfine_m.TXT")
	dataframe_after_CV = pd.read_csv("20230224_2-6_XRD_afterCV60scans.TXT")

	#PDFデータ
	data_frame_PDF = pd.read_csv("PDF_card_alphaMnO2_00-044-0141.csv")

	# 描画範囲設定
	fig = plt.figure(figsize = (4,3))
	ax = fig. add_axes([0.1,0.2,0.8,0.7])
	ax.set_xlabel(r'2$\theta$ [degree]')
	ax.set_ylabel('Intensity [a.u.]')
	X_MIN: float = 5
	X_MAX: float = 50
	Y_MIN: float = -700
	Y_MAX: float = 2700
	ax.set_xlim(X_MIN, X_MAX)
	ax.set_ylim(Y_MIN, Y_MAX)
	ax.tick_params(labelleft=False, left=False, right=False)

	#プロット
	y_offset_before = 600
	y_offset_after = 1600
	y_offset_PDF = -400
	delta_yoffset_text = 700

	ax.axhline(y_offset_before, xmin=0, xmax=1, c="k", lw=0.5)
	ax.plot(
		dataframe_before_CV['2theta'],
		dataframe_before_CV['intensity']*0.7+y_offset_before,
		c='#FF3333',
		lw= 1,
		)

	ax.axhline(y_offset_after, xmin=0, xmax=1, c="k", lw=0.5)
	ax.plot(
		dataframe_after_CV['2theta'],
		dataframe_after_CV['intensity']*0.6+y_offset_after,
		c='#3333FF',
		lw= 1,
		)
	
	plot_PDF(ax, data_frame_PDF, y_offset_PDF, (Y_MIN, Y_MAX), 0.0015,"#FF3333")
	
	#annotations
	ax.text(
		6, y_offset_PDF+delta_yoffset_text,
		"PDF 00-044-0141",
		c="#FF3333",
		fontsize=10
	)

	ax.text(
		6, y_offset_before+delta_yoffset_text,
		"before 60 CV scans",
		c="#FF3333",
		fontsize=10
	)

	ax.text(
		6, y_offset_after+delta_yoffset_text,
		"after 60 CV scans",
		c="#3333FF",
		fontsize=10
	)

	fig.savefig("XRD_before_after_CV.png", dpi=600)

	fig = plt.figure(figsize = (4,3))
	ax = fig. add_axes([0.1,0.2,0.8,0.7])
	ax.set_xlabel(r'2$\theta$ [degree]')
	ax.set_ylabel('Intensity [a.u.]')
	X_MIN: float = 5
	X_MAX: float = 50
	Y_MIN: float = -0.2
	Y_MAX: float = 0.2
	ax.set_xlim(X_MIN, X_MAX)
	ax.set_ylim(Y_MIN, Y_MAX)
	ax.tick_params(labelleft=False, left=False, right=False)

	before = dataframe_before_CV['intensity'].values
	after = dataframe_after_CV['intensity'].values
	ax.plot(
		dataframe_after_CV['2theta'],
		(after-np.min(after))/np.max(after)-\
		(before - np.min(before))/np.max(before)*0.45,
		c='#3333FF',
		lw= 1,
		)
	
	fig.savefig("XRD_diff.png", dpi=600)
	
	return

PXRD_after_CVs()
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""


# %%