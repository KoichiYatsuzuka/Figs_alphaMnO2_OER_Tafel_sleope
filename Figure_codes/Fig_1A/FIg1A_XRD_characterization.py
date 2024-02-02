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

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')

# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import F, R, T
from Common_for_figs import plot_PDF
os.chdir(Path(__file__).resolve().parent.__str__())
Common_for_figs.set_common_matlptlib_rcParameter()

"""Material characterization
***XRD_MnO2.png
"""
def PXRD_MnO2_powder():
	# 実測値読み込み
	data_frame_powder = pd.read_csv("20230315_2-7_003_pxrd_non-sprayed_wide.TXT")
	data_frame_FTO = pd.read_csv("20230224_2-6_BareFTO_superfine.TXT")
	data_frame_sprayed = pd.read_csv("20230112_2-4_006_alpha-MnO2_CV-sweeped_superfine_m.TXT")

	# PDF情報読み込み
	data_frame_PDF = pd.read_csv("PDF_card_alphaMnO2_00-044-0141.csv")

	fig = plt.figure(figsize = (4,3))
	#ax = fig.add_axes([0.1,0.2,0.8,0.7])
	
	BOTTOM_MARGIN = 0.15

	ax_powder 	= fig.add_axes([0.1,BOTTOM_MARGIN + 0.6,0.8,0.2])
	ax_sprayed 	= fig.add_axes([0.1,BOTTOM_MARGIN + 0.4,0.8,0.2])
	ax_FTO 		= fig.add_axes([0.1,BOTTOM_MARGIN + 0.2,0.8,0.2])
	ax_PDF 		= fig.add_axes([0.1,BOTTOM_MARGIN + 0.0,0.8,0.2])
	all_axes = [
		ax_powder, ax_sprayed, ax_FTO, ax_PDF
	]
	
	# 描画範囲設定
	ax_PDF.set_xlabel(r'2$\theta$ [degree]')
	ax_FTO.set_ylabel('            Intensity [a.u.]') #idiot
	
	X_MIN = 5
	X_MAX = 45
	Y_MIN = -50
	Y_MAX = 1000

	for ax in all_axes:
		ax.set_xlim(X_MIN, X_MAX)
		ax.set_ylim(Y_MIN,Y_MAX)
		ax.tick_params(labelleft=False, left=False, right=False, top=False)
		
		
	ax_PDF.set_ylim(Y_MIN, Y_MAX)
	
	# プロット
	## プロット関連変数設定

	#色
	red = "#FF3333"
	purple = '#AA33AA'
	blue = "#3333FF"

	## ただの粉体
	ax_powder.plot(
		data_frame_powder['2theta'],
		data_frame_powder['intensity']*3,
		c= red,
		lw= 1,
		)
	
	## spray coating後
	ax_sprayed.plot(
		data_frame_sprayed['2theta'],
		data_frame_sprayed['intensity']*1.5,
		c=purple,
		lw= 1,
		)

	## FTOのみ
	ax_FTO.plot(
		data_frame_FTO['2theta'],
		data_frame_FTO['intensity']/1.2,
		c=blue,
		lw= 1,
		label="FTO substrate" 
		)
	
	## PDFのピークトップの縦線を入れる
	plot_PDF(
		ax_PDF,
		data_frame_PDF,
		y_offset=100,
		y_lims=(Y_MIN, Y_MAX),
		y_scale=0.008,
		color="r"
	)

	# anotations
	ax_powder.text(2, Y_MAX, "(A)")

	y_deltaoffset_text=+730
	fs = 10 # font size

	ax_FTO.text(5.5, y_deltaoffset_text, "bare FTO substrate", c=blue, fontsize=fs)
	ax_sprayed.text(5.5, y_deltaoffset_text, r"$\alpha$-MnO$_2$ on FTO", c=purple, fontsize=fs)
	ax_powder.text(5.5, y_deltaoffset_text, r"$\alpha$-MnO$_2$", c=red, fontsize=fs)
	ax_PDF.text(5.5, y_deltaoffset_text, "PDF 00-044-0141", c=red, fontsize=fs)

	plt.savefig("XRD_MnO2.png", dpi = 600)

PXRD_MnO2_powder()

""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""

# %%
