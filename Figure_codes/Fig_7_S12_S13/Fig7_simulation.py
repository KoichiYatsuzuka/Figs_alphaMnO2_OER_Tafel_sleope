#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from mpl_toolkits.mplot3d import axes3d

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import F, R, T, ELECTRON_NUM_OER, SSCE_TO_SHE
from Common_for_figs import color_array
from Common_for_figs import color_map_RGB,color_map_KtoR
from Common_for_figs import (
    ScanNumber,
    SimdVoltammogram,
    ExperimentalVoltammogram,
    SimdVoltammogramList,
    ExpVoltammogramList,
    RedoxKinetics,
    RedoxKineticsArray

)
os.chdir(Path(__file__).resolve().parent.__str__())

Common_for_figs.set_common_matlptlib_rcParameter()


"""数値シミュレーションによるTrumpet plotパラメータからのボルタモグラム予測と
実測との比較
"""


"""------------variable parameters-------------------"""
n = 4
"""the number of electron required for OER"""
N   = 8E-10
"""the density of active site [1/cm2]"""
E2  = 0.91
"""redox potential of Mn(IV)/Mn(III) [V vs SHE]"""
alpha2 = 0.5
"""electron transfer coefficient of Mn(IV)/Mn(III)"""
k20 = 4E4
"""standard rate constant of Mn(IV)/Mn(III) [1/s]"""
j_baseline = 0.13
"""non-Faraday current density[mA/cm2]"""
nFN = n*F*N
rds_redox_kinetics_ = RedoxKinetics(alpha2, 1, k20, E2)
"""--------------------------------------------------"""
E = np.linspace(0.65, 1.0, 201)
datapoint_num: int = 25
"""trumpet plotのデータの何個を使うか"""
raw_cv_cycle_num = [
    ScanNumber(1),
    ScanNumber(9),
    ScanNumber(60)
]
"""cv cycles of interst"""

# 実測データの読み込み
# 一部のデータは最後の描画でも使う
def load_redox_kinetics_params():

    #trumpet plotの解析データ
    experimental_redox_data_frame = pd.read_csv('../Fig_3-5/Redox_parameters_from_trumpet.csv')
    tmp_scannum = []
    tmp_redox_params = []
    for index in range(len(experimental_redox_data_frame['index'].values)):
        tmp_scannum.append(ScanNumber(experimental_redox_data_frame['index'][index]))
        tmp_redox_params.append(RedoxKinetics(
            experimental_redox_data_frame['alpha'][index],
            experimental_redox_data_frame['n'][index],
            experimental_redox_data_frame['k'][index],
            experimental_redox_data_frame['E0'][index]
        ))
    experimental_redox_kinetics = RedoxKineticsArray(tmp_scannum, tmp_redox_params)

    return experimental_redox_kinetics

def load_experimental_voltammogram(
    cv_cycle_num_list: list[ScanNumber], 
    df_actual_voltammogram_data: pd.DataFrame
    ):

    df = df_actual_voltammogram_data

    raw_cv_files = []
    for num in cv_cycle_num_list:
        row = df[df["cycle"] == num.__int__()]
        file_name: str = row["file name"].values[0]
        raw_cv_files.append(copy.deepcopy(file_name))

    cv_list = [] 
    for file_name in raw_cv_files:
        df_tmp = pd.read_csv("../Fig_3-5/"+file_name)
        cv_list.append(ExperimentalVoltammogram(copy.deepcopy(df_tmp)))
    
    exp_cv_list = ExpVoltammogramList(cv_cycle_num_list,cv_list)

    return exp_cv_list

def simulate_voltammogram(
        pre_redox_kinetics: RedoxKinetics,
        rds_redox_kinetics: RedoxKinetics,
        potential_range: np.ndarray,
        j_baseline: float,
        relative_surface_area: float = 0.5
    )->SimdVoltammogram:

    k1 = pre_redox_kinetics.forward_rate_array(potential_range)
    k1r = pre_redox_kinetics.backward_rate_array(potential_range)
    k2 = rds_redox_kinetics.forward_rate_array(potential_range)
    k2r = rds_redox_kinetics.backward_rate_array(potential_range)

    k_total = (k1*k2-k1r*k2r)/(k1+k1r+k2+k2r)
    j_faraday = ELECTRON_NUM_OER * F * relative_surface_area * N * k_total

    j_total = j_faraday + j_baseline

    simd_voltammogram = SimdVoltammogram(copy.copy(potential_range), j_total)

    return simd_voltammogram

def draw_Tafel_plot(
        experimental_cvs: ExpVoltammogramList, 
        simulated_cvs: SimdVoltammogramList
        ):
    """Overlays experimental and simulated Tafel plot.
    """

    # 描画範囲設定
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel('$E$ [V vs. SHE]')
    ax.set_ylabel('log $j$ [mA/cm$^2$]')
    ax.set_xlim(0.68,1.0)
    ax.set_ylim(-1, 1.8)

    # 描画
    ## experimental Tafel plots
    for index, exp_scan_num in enumerate(experimental_cvs.scan_nums):
        exp_cv_data = experimental_cvs[exp_scan_num]
        
        potential = exp_cv_data.corrected_potential[:exp_cv_data.anodic_edge] + SSCE_TO_SHE
        """Ag/AgCl -> SHE"""
        log_j = (np.log10(exp_cv_data.current_mA[:exp_cv_data.anodic_edge]/2))
        """log(I in A) -> log (j in mA/cm2)"""

        ax.plot(potential, log_j, c = color_array[index], lw=1.0)

    ## simulated Tafel plots
    for index, exp_scan_num in enumerate(experimental_cvs.scan_nums):
        simd_cv_data = simulated_cvs[exp_scan_num]

        potential = simd_cv_data.potential
        log_j = simd_cv_data.log_current

        ax.plot(potential, log_j, c= color_array[index], linestyle=":", lw=1.5)

    plt.savefig('Sim_experimental_and_simulated_Tafel_plot.png', dpi=600)
    return 

def draw_Tafel_plot_only_Faraday(
    experimental_cvs: ExpVoltammogramList, 
    simulated_cvs_only_Faraday: SimdVoltammogramList
):
    #-----------------------------------------------------------------------------
    #------------------------充電電流なしのTafel slope----------------------------
    #-----------------------------------------------------------------------------
    #充電電流なしのFaraday電流のみのsimulation


    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel('$E$ [V vs. SHE]')
    ax.set_ylabel('Tafel slope [mV/dec]')
    ax.set_xlim(0.75,0.95)
    ax.set_ylim(0, 200)

    ## experimental
    for index, exp_scan_num in enumerate(experimental_cvs.scan_nums):
        exp_cv_data = experimental_cvs[exp_scan_num]
        
        potential = exp_cv_data.corrected_potential[:exp_cv_data.anodic_edge] + SSCE_TO_SHE
        """Ag/AgCl -> SHE"""

        Tafel = exp_cv_data.Tafel_slope[:exp_cv_data.anodic_edge]

        ax.plot(potential, Tafel, c=color_array[index], lw=1.0)

    ## simulated
    for index, exp_scan_num in enumerate(experimental_cvs.scan_nums):
        simd_cv_data = simulated_cvs_only_Faraday[exp_scan_num]

        potential = simd_cv_data.potential
        Tafel = simd_cv_data.Tafel_slope

        ax.plot(potential, Tafel, c= color_array[index], linestyle=":", lw=1.5)

    plt.savefig("Sim_Tafel_slope_Faraday.png", dpi=600)
    return 

def draw_Tafel_slopes(
    experimental_cvs: ExpVoltammogramList, 
    simulated_cvs: SimdVoltammogramList
):
    #-----------------------------------------------------------------------------
    #--------------実測Tafelとsimulated Tafelの比較-----------------------
    #-----------------------------------------------------------------------------
    # 描画範囲設定
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel('$E$ [V vs. SHE]')
    ax.set_ylabel('Tafel slope [mV/dec]')
    ax.set_xlim(0.75,0.95)
    ax.set_ylim(0, 200)

    ## experimental
    for index, exp_scan_num in enumerate(experimental_cvs.scan_nums):
        exp_cv_data = experimental_cvs[exp_scan_num]
        
        potential = exp_cv_data.corrected_potential[:exp_cv_data.anodic_edge] + SSCE_TO_SHE
        """Ag/AgCl -> SHE"""

        Tafel = exp_cv_data.Tafel_slope[:exp_cv_data.anodic_edge]

        ax.plot(potential, Tafel, c=color_array[index], lw=1.0)

    ## simulated
    for index, exp_scan_num in enumerate(experimental_cvs.scan_nums):
        simd_cv_data = simulated_cvs[exp_scan_num]

        potential = simd_cv_data.potential
        Tafel = simd_cv_data.Tafel_slope

        ax.plot(potential, Tafel, c= color_array[index], linestyle=":", lw=1.5)

    plt.savefig('Sim_experimental_and_simulated_Tafel slope.png', dpi=600)
    return

def draw_Tafel_slope_parity(
    actual_minimum_Tafel_slopes: np.ndarray,
    simulated_cvs: SimdVoltammogramList
):
    #-----------------------------------------------------------------------------
    #--------------------------parity plot-Tafel slope----------------------------
    #-----------------------------------------------------------------------------

    # calculate alpha_eff from the simulations
    tmp: list =[]
    for index, scan_num in enumerate(simulated_cvs.scan_nums):
        cv = simulated_cvs[scan_num]
        min_Tafel_slope = np.min(cv.Tafel_slope)
        tmp.append(min_Tafel_slope)
    #simulated_alpha_eff = np.array(tmp)
    simulated_Tafel:np.ndarray = np.array(tmp)

    #r^2計算
    ## 残差の平方和
    sum_sq_residual = ((actual_minimum_Tafel_slopes-simulated_Tafel)**2).sum()

    ## 全変動
    avr_actual_Tafel = actual_minimum_Tafel_slopes.sum()/actual_minimum_Tafel_slopes.size
    total_variation = ((actual_minimum_Tafel_slopes- avr_actual_Tafel)**2).sum()

    r_sq = 1-(sum_sq_residual/total_variation)

    print("Tafel party  r^2 = ", r_sq)

    #描画範囲設定
    fig = plt.figure(figsize = (3,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel('Experimental Tafel slope [mV/dec]')
    ax.set_ylabel('Simulated Tafel slope [mV/dec]')
    ax.set_xlim(55,80)
    ax.set_xticks(np.linspace(55, 80, 6))
    ax.set_ylim(55,80)

    #プロット
    ax.scatter(actual_minimum_Tafel_slopes, simulated_Tafel, c="#FF3030")
    ax.plot(np.linspace(0.0, 150, 10), np.linspace(0.0, 150, 10), 'k--')
    plt.savefig("Sim_Tafel_vs_actual_Tafel", dpi=600)
    
    return

def draw_k1_k2_vs_scan(
    k1_redox_kinetics_data: RedoxKineticsArray,
    k2_redox_kinetics_data: RedoxKinetics    
):
    applied_potential=0.9
    scans_raw = k1_redox_kinetics_data.scan_nums
    scans = []
    for scan_raw in scans_raw:
        scans.append(scan_raw.scan_number)

    k1_ary = []
    for scan in k1_redox_kinetics_data.scan_nums:

        k1_data = k1_redox_kinetics_data[scan]

        k1_ary.append(k1_data.forward_rate(applied_potential))
    #k1_redox_kinetics_data.k0_values*\
    #            np.exp(k1_redox_kinetics_data.alpha_values*F*(applied_potential-k1_redox_kinetics_data.E0_values))

    k2_ary = []
    for i in range(0, len(scans)):
        k2_ary.append(k2_redox_kinetics_data.forward_rate(applied_potential))

    # 描画範囲設定
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel('CV cycles')
    ax.set_ylabel('log (Reaction rate [1/s])')
    ax.set_xlim(-1,61)
    #ax.set_ylim(0, 200)

    ax.scatter(
        scans,
        np.log10(k1_ary),
        c="r",
        label="$k_1$"
    )

    ax.scatter(
        scans,
        np.log10(k2_ary),
        c="b",
        label="$k_2$"
    )

    ax.legend()


    plt.savefig("calcd-k1_opt-k2_vs_cycle.png", dpi=600)

    return
    
    

def numerical_simulations():

    experimental_redox_kinetics = load_redox_kinetics_params()

    #各掃引ごとにalpha_eff最大値などを解析したデータ
    actual_voltammogram_data_frame = \
        pd.read_csv('../Fig_3-5/Voltammogram_alpha_eff_data.csv')
    actual_alpha_eff:np.ndarray = \
        actual_voltammogram_data_frame['max alpha_eff'][:datapoint_num].values 
    actual_Tafel:np.ndarray \
        = (actual_voltammogram_data_frame['minimum Tafel slope'][:datapoint_num].values)*1000 
        #V/dec -> mV/dec
    actual_normalized_charge:np.ndarray =\
        actual_voltammogram_data_frame['normalized charge'].values[:datapoint_num]

    experimental_cvs = load_experimental_voltammogram(raw_cv_cycle_num, actual_voltammogram_data_frame)

    # simulate voltammograms with the experimental parameters 
    tmp_list: list[SimdVoltammogram] =[]
    for index, scan_num in enumerate(experimental_redox_kinetics.scan_nums):
        tmp_list.append(
            simulate_voltammogram(
                experimental_redox_kinetics[scan_num],
                rds_redox_kinetics=rds_redox_kinetics_,
                potential_range=E,
                j_baseline = j_baseline,
                relative_surface_area=actual_normalized_charge[index]
            )
        )
    simulated_cvs = \
        SimdVoltammogramList(
            experimental_redox_kinetics.scan_nums,
            copy.deepcopy(tmp_list)
            )

    # simulate voltammograms without non-Faraday current
    tmp_list: list[SimdVoltammogram] =[]
    for index, scan_num in enumerate(experimental_redox_kinetics.scan_nums):
        tmp_list.append(
            simulate_voltammogram(
                experimental_redox_kinetics[scan_num],
                rds_redox_kinetics=rds_redox_kinetics_,
                potential_range=E,
                j_baseline = 0,
                relative_surface_area=actual_normalized_charge[index]
            )
        )
    simulated_cvs_only_Faraday = \
        SimdVoltammogramList(
            experimental_redox_kinetics.scan_nums,
            copy.deepcopy(tmp_list)
            )
        
    draw_Tafel_plot(
        experimental_cvs,
        simulated_cvs
    )

    draw_Tafel_plot_only_Faraday(
        experimental_cvs,
        simulated_cvs_only_Faraday
    )

    draw_Tafel_slopes(
        experimental_cvs,
        simulated_cvs
    )

    draw_Tafel_slope_parity(
        actual_Tafel,
        simulated_cvs
    )

    #論文には使われていないが学会用に
    draw_k1_k2_vs_scan(
        experimental_redox_kinetics,
        rds_redox_kinetics_
    )

    return

numerical_simulations()

""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""
""":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"""

# %%
