#%%
# 共通部分の定義読み込み
#import common_for_figs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
from copy import deepcopy
from dataclasses import dataclass

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
from BFC_libs import Raman as rmn
import BFC_libs.common as common
import BFC_libs.Raman as rmn 
os.chdir(Path(__file__).resolve().parent.__str__())

common.set_matpltlib_rcParameters()
# Ramman peak shift during CVs


def spectral_normal_deviation(spectra: dict[str, rmn.RamanSpectrum]):
    spectral_names = list(spectra.keys())
    wavenumber_series = spectra[spectral_names[0]].wavenumber

    # 返り値用
    std_dev_array = np.array([])

    for index in range(len(wavenumber_series)):
        
        # 各indexのデータポイントを集める
        data_point_list = []
        for data_name in spectral_names:
            data_point_list.append(spectra[data_name].intensity[index])

        # 集めたデータポイントの標準偏差を求める
        std_dev = np.std(data_point_list)

        std_dev_array=np.append(std_dev_array, std_dev)

    return deepcopy(std_dev_array)


def raman_peak_shift_during_CV():
 
    dir = "20230928_2-54_BfrAftCV_finely-calibrated"

    column_name_base = "N_{}CV"
    CV_scans = ["0", "15", "30", "45", "60"]
    column_names =[]
    for CV_scan in CV_scans:
        column_names.append(column_name_base.format(CV_scan))

    #df_550_610 = rmn.read_1D_data(dir+"/comp_normd550-610.txt")
    #df_600_650 = rmn.read_1D_data(dir+"/comp_normd600-650.txt")
    df_550_675 = rmn.read_1D_data(dir+"/comparison.txt")

    df_raw_data_0CV = rmn.read_1D_data(dir+"/0CV_raw.txt")
    df_raw_data_60CV = rmn.read_1D_data(dir+"/60CV_raw.txt")

    df_normd_data_0CV = {}
    for spectrum_name in df_raw_data_0CV.keys():
        rawspc = deepcopy(df_raw_data_0CV[spectrum_name])
        base = rawspc.intensity.min()
        scale = rawspc.intensity.max()-rawspc.intensity.min()
        normdspc = rmn.RamanSpectrum(
            rawspc.wavenumber,
            (rawspc.intensity-base)/scale,
            rawspc.data_name
        )
        df_normd_data_0CV[spectrum_name] = normdspc
    
    df_normd_data_60CV = {}
    for spectrum_name in df_raw_data_60CV.keys():
        rawspc = deepcopy(df_raw_data_60CV[spectrum_name])
        base = rawspc.intensity.min()
        scale = rawspc.intensity.max()-rawspc.intensity.min()
        normdspc = rmn.RamanSpectrum(
            rawspc.wavenumber,
            (rawspc.intensity-base)/scale,
            rawspc.data_name
        )
        df_normd_data_60CV[spectrum_name] = normdspc


    std_deviation_0CV_raw = spectral_normal_deviation(df_normd_data_0CV)
    std_deviation_60CV_raw = spectral_normal_deviation(df_normd_data_60CV)

    spectra_magnification_ratio_0CV:float = \
        df_550_675["N_0CV"].intensity.max() - df_550_675["N_0CV"].intensity.min()

    spectra_magnification_ratio_60CV:float = \
        df_550_675["N_60CV"].intensity.max() - df_550_675["N_60CV"].intensity.min()

    std_deviation_0CV_adj = std_deviation_0CV_raw/spectra_magnification_ratio_0CV
    std_deviation_60CV_adj = std_deviation_60CV_raw/spectra_magnification_ratio_60CV

    # loading peak fit result from a text file
    df_fit = pd.read_csv(dir+"/fitting_result_500-700.txt", sep="\t")
    df_fit_each_scan = []
    for i in range(0, 5):
        df_fit_each_scan.append(df_fit[:][0+i*4:4+i*4])

    @dataclass
    class InsetRange:
        x_begin: float
        x_end: float
        y_begin: float
        y_end: float

    #def draw_spectra(inset_range: InsetRange, peak_list: ):

    # drawing overall spectra + 630 cm-1 peak
    fig1, ax_spectra = common.create_standard_matplt_canvas()
    ax_spectra.tick_params(left = False, right=False, labelleft=False)
    ax_spectra.text(
        440,
        1.03,
        "(A)"
    )
    ax_spectra.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax_spectra.set_ylabel("Normalized intensity [a.u.] ")
    ax_spectra.set_xlim(475,675)
    ax_spectra.set_ylim(0.2,1.05)

    ax_spectra.plot(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_0CV"].intensity,
        c="r",
        lw=0.5
    )
    ax_spectra.plot(
        df_550_675["N_60CV"].wavenumber,
        df_550_675["N_60CV"].intensity,
        c="b",
        lw=0.5
    )

    

    inset_range =InsetRange(622, 642, 0.75, 0.85)

    ax_spectra.add_patch(
        Rectangle(
            xy= (inset_range.x_begin, inset_range.y_begin),
            width=inset_range.x_end-inset_range.x_begin,
            height=inset_range.y_end-inset_range.y_begin,
            ls="--",
            fc="none",
            ec="k"
        )
    )

    ax_inset:plt.Axes = inset_axes(
        ax_spectra, 
        width=1.0, 
        height=0.9, 
        loc=4,
        bbox_to_anchor = (0.01, 0.45, 0.4, 0.9),
        bbox_transform = ax_spectra.transAxes)

    ax_inset.tick_params(
        labelleft=False,
        left=False,
        right=False
    )

    ax_inset.set_xlim(inset_range.x_begin, inset_range.x_end)
    ax_inset.set_ylim(inset_range.y_begin, inset_range.y_end)

    # 先に誤差のfillをすることで、線グラフが覆われるのを防ぐ
    ax_inset.fill_between(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_0CV"].intensity+std_deviation_0CV_adj,
        df_550_675["N_0CV"].intensity-std_deviation_0CV_adj,
        color="#FF8888",
        alpha=0.3
    )
    ax_inset.fill_between(
        df_550_675["N_60CV"].wavenumber,
        df_550_675["N_60CV"].intensity+std_deviation_60CV_adj,
        df_550_675["N_60CV"].intensity-std_deviation_60CV_adj,
        color="#8888FF",
        alpha=0.3
    )

    ax_inset.plot(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_0CV"].intensity,
        c="r",
        lw=0.5
    )
    ax_inset.plot(
        df_550_675["N_60CV"].wavenumber,
        df_550_675["N_60CV"].intensity,
        c="b",
        lw=0.5
    )

    plt.savefig("Raman_spectra_shift_630.png", dpi=600)

    # drawing peak fit results at 630 cm-1
    fig3, ax_scatter = common.create_standard_matplt_canvas()
    ax_scatter.set_xlabel("Cycle numbers")
    ax_scatter.set_xticks([0, 15, 30, 45, 60])
    ax_scatter.set_ylabel("Peak position [cm$^{-1}$]")
    for i, df_separated in enumerate(df_fit_each_scan):
        ax_scatter.scatter(

            int(CV_scans[i]),
            df_separated["Position"][1+4*i],
            c="r"
        )
    ax_scatter.text(
        -13.5,
        633.6,
        "(B)"
        )
    plt.savefig("Raman_peak_top_shift_630.png", dpi=600)

    #-----------------------------------------------------

    fig1, ax_spectra = common.create_standard_matplt_canvas()
    ax_spectra.text(
        440,
        1.03,
        "(A)"
    )
    ax_spectra.tick_params(left = False, right=False, labelleft=False)
    ax_spectra.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax_spectra.set_ylabel("Normalized intensity [a.u.] ")
    ax_spectra.set_xlim(475,675)
    ax_spectra.set_ylim(0.2,1.05)

    """for i, column_name in enumerate(column_names):
        ax_spectra.plot(
            df_550_675["wn(N_N_0CV)"],
            df_550_675[column_name],
            c=colors.color_map_KtoR(i/len(column_names)),
            lw=0.5
        )"""
    ax_spectra.plot(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_0CV"].intensity,
        c="r",
        lw=0.5
    )
    ax_spectra.plot(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_60CV"].intensity,
        c="b",
        lw=0.5
    )

    
    
    inset_range =InsetRange(577, 586, 0.9, 1.01)

    ax_spectra.add_patch(
        Rectangle(
            xy= (inset_range.x_begin, inset_range.y_begin),
            width=inset_range.x_end-inset_range.x_begin,
            height=inset_range.y_end-inset_range.y_begin,
            ls="--",
            fc="none",
            ec="k"
        )
    )

    ax_inset:plt.Axes = inset_axes(
        ax_spectra, 
        width=1.0, 
        height=0.9, 
        loc=4,
        bbox_to_anchor = (0.01, 0.45, 0.4, 0.9),
        bbox_transform = ax_spectra.transAxes)

    ax_inset.tick_params(
        labelleft=False,
        left=False,
        right=False
    )

    ax_inset.set_xlim(inset_range.x_begin, inset_range.x_end)
    ax_inset.set_ylim(inset_range.y_begin, inset_range.y_end)

    # 先に誤差のfillをすることで、線グラフが覆われるのを防ぐ
    ax_inset.fill_between(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_0CV"].intensity+std_deviation_0CV_adj,
        df_550_675["N_0CV"].intensity-std_deviation_0CV_adj,
        color="#FF8888",
        alpha=0.3
    )
    ax_inset.fill_between(
        df_550_675["N_60CV"].wavenumber,
        df_550_675["N_60CV"].intensity+std_deviation_60CV_adj,
        df_550_675["N_60CV"].intensity-std_deviation_60CV_adj,
        color="#8888FF",
        alpha=0.3
    )

    ax_inset.plot(
        df_550_675["N_0CV"].wavenumber,
        df_550_675["N_0CV"].intensity,
        c="r",
        lw=0.5
    )

    ax_inset.plot(
        df_550_675["N_60CV"].wavenumber,
        df_550_675["N_60CV"].intensity,
        c="b",
        lw=0.5
    )

    plt.savefig("Raman_spectra_shift_580.png", dpi=600)

    fig3, ax_scatter = common.create_standard_matplt_canvas()
    ax_scatter.set_xlabel("Cycle numbers")
    ax_scatter.set_xticks([0, 15, 30, 45, 60])
    ax_scatter.set_ylabel("Peak position [cm$^{-1}$]")
    for i, df_separated in enumerate(df_fit_each_scan):
        ax_scatter.scatter(

            int(CV_scans[i]),
            df_separated["Position"][0+4*i],
            c="r"
        )
    ax_scatter.text(
        -13.5,
        581.1,
        "(B)"
        )
    plt.savefig("Raman_peak_top_shift_580.png", dpi=600)

raman_peak_shift_during_CV()

# %%
