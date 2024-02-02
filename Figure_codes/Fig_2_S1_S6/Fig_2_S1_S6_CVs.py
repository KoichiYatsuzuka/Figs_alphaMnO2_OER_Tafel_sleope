"""
実測ボルタモグラム
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
from collections import namedtuple

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import color_map_RGB, color_map_RtoB
from Common_for_figs import SSCE_TO_SHE
Common_for_figs.set_common_matlptlib_rcParameter()
os.chdir(Path(__file__).resolve().parent.__str__())

""" EC_CVs, EC_CVs_no_iR, EC_onsetpotential.png
*** 普通のCV, 拡大CV, Tafelみたいなやつの重ね書き
*** 大き目のキャンバスの上部に普通のCV、左下に拡大図、右下にlog I vs Eのグラフ
"""
def CVs():
    # ----------CONST values required to change----------------------
    #どのファイルの番号がどの掃引速度に対応するかのtuple
    FILE_NUM_LIST: list =[1, 10, 20, 30, 40, 50, 60]
    # ファイル名の共通部分(.formatで数値を入れる)
    FILE_NAME_BASE: str = '20220924_003_MultiCV_0_850mV_1-84_C_02_CV_C01_{}.csv'

    #置換予定
    class VoltammogramData:
        cycle_num: int = 0 #何回CVを掃引したときのデータか
        recorded_potential: list =[] #ポテスタで記録される生の電位
        corrected_potential: list = [] #iR補正をした電位
        corrected_potential_anodic :list =[] #iR補正の値のうち、最初の半分（おのずとanodic sweep分）
        current_density_mA: list = [] #電流密度（mA/cm2）
        log_current_density: list = [] #log_10 (current [A/cm2])

        def __init__(
            self,
            cycle_num: int ,
            recorded_potential: list ,
            corrected_potential: list,
            corrected_potential_anodic :list ,
            current_density_mA: list ,
            log_current_density: list,
        ):
            self.cycle_num = cycle_num
            self.recorded_potential = recorded_potential
            self.corrected_potential = corrected_potential
            self.corrected_potential_anodic = corrected_potential_anodic
            self.current_density_mA = current_density_mA
            self.log_current_density = log_current_density

    voltammograms: list[VoltammogramData] = []

    # End of defintions of consts and classes
    # Procedure starts

    for file_num in FILE_NUM_LIST:
        file_name=FILE_NAME_BASE.format(file_num)
        data_file = pd.read_csv(file_name)
        
        anodic_sweep_len = int(len(data_file['Ewe/V'])/2) #半分の長さのlistを作成するため
        
        voltammogram = VoltammogramData(
            cycle_num= file_num,
            recorded_potential= data_file['Ewe/V'].values + SSCE_TO_SHE, 
            corrected_potential= data_file['E-IR/V'].values+ SSCE_TO_SHE, 
            current_density_mA= data_file['<I>/mA'].values/2, # mA -> mA/cm2
            log_current_density= np.log10(abs(data_file['<I>/mA']/2))[:anodic_sweep_len].values,
            corrected_potential_anodic= data_file['E-IR/V'][:anodic_sweep_len].values + SSCE_TO_SHE
        )

        voltammograms.append(voltammogram)


    fig = plt.figure(figsize = (4,4))

    #図のサイズ調整用
    LEFT_SPACE = 0.2
    SPACE_HORIZONTAL = 0.08
    SPACE_VERTICAL = 0.2
    WIDTH_BOTTOM = 0.3

    ax1 = fig.add_axes([0.15, 0.52+SPACE_HORIZONTAL, WIDTH_BOTTOM*2+SPACE_VERTICAL, 0.32])
    ax2 = fig.add_axes([0.15, 0.12, WIDTH_BOTTOM, 0.32])
    ax3 = fig.add_axes([0.15+WIDTH_BOTTOM+SPACE_VERTICAL, 0.12, WIDTH_BOTTOM, 0.32])

    ax1.set_xlabel('$E$ [V vs. SHE]')
    ax1.set_ylabel('$j$ [mA/cm$^2$]')
    ax2.set_xlabel('$E$ [V vs. SHE]')
    ax2.set_ylabel('$j$ [mA/cm$^2$]')
    ax3.set_xlabel('$E$ [V vs. SHE]')
    ax3.set_ylabel('log$_{10}$ $j$ [mA/cm$^2$]')

    ax1.set_xlim(0.2, 0.85 + 0.2)
    ax2.set_xlim(0.2, 0.85*WIDTH_BOTTOM/(WIDTH_BOTTOM*2+SPACE_VERTICAL)+ 0.2)
    ax3.set_xlim(0.85*(WIDTH_BOTTOM+SPACE_VERTICAL)/(WIDTH_BOTTOM*2+SPACE_VERTICAL)+ 0.2, 0.85+ 0.2)

    ax1.set_ylim(-1.5, 12.5)
    ax2.set_ylim(-1.1, 1.1)
    ax3.set_ylim(-0.8, 1.2)

    #プロットのパラメーター
    LINE_WIDTH = 0.75

    for i, voltammogram in enumerate(voltammograms):
        color = color_map_RGB(i/len(voltammograms))

        ax1.plot(
            voltammogram.corrected_potential,
            voltammogram.current_density_mA,
            lw=LINE_WIDTH,
            c=color
        )

        ax2.plot(
            voltammogram.corrected_potential ,
            voltammogram.current_density_mA ,
            lw=LINE_WIDTH,
            c=color
        )

        ax3.plot(
            voltammogram.corrected_potential_anodic ,
            voltammogram.log_current_density,
            lw=LINE_WIDTH,
            c=color
        )

    ax1.axhline(1, c='k', lw=0.5, linestyle="--")
    ax3.axhline(0.0, c='k', lw=0.5, linestyle="--")

    ax1.text(
        -0.1, 1,
        "(A)",
        transform= ax1.transAxes
    )

    ax2.text(
        -0.3, 1,
        "(B)",
        transform= ax2.transAxes
    )

    ax3.text(
        -0.3, 1,
        "(C)",
        transform= ax3.transAxes
    )

    plt.savefig("EC_CVs", dpi=600)

    # iR補正前
    fig = plt.figure(figsize = (4,3))
    ax = fig. add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel('$E$ [V vs. SHE]')
    ax.set_ylabel('$j$ [mA/cm$^2$]')
    ax.set_xlim(0.2, 1.06)
    for i, voltammogram in enumerate(voltammograms):
        ax.plot(
            voltammogram.recorded_potential,
            voltammogram.current_density_mA,
            lw=LINE_WIDTH,
            c=color_map_RGB(i/len(voltammograms))
        )
    plt.savefig("EC_CVs_no_iR", dpi=600)

    #--------------------------------------------------------
    # onset potentialリスト
    #--------------------------------------------------------
    log_j_list = [
         -0.5, -0.25, 0, 0.25, 0.5
    ]
    # 型定義
    OnsetPotential = namedtuple('OnsetPotential', 'log_j onset_potential')
    ## OP = onset potential
    OPEachScan = namedtuple('OP_each_scan', 'scan OP')
    
    # CVデータからonset potentialの情報を抜き出してlist化
    OP_each_scan = []
    for cv_index, cv in enumerate(voltammograms):
        OP_list = []
        separating_potential = 0.8
        
        i = np.argmin(np.abs(
                cv.corrected_potential_anodic - separating_potential
            ))
        potential_catalysis = cv.corrected_potential_anodic[i:]
        Tafel_plot = cv.log_current_density[i:]
        for j_index,log_j in enumerate (log_j_list):
            k = np.argmin(np.abs(
                Tafel_plot - log_j
            ))
            OP_list.append(OnsetPotential(log_j, potential_catalysis[k]))
        OP_each_scan.append(OPEachScan(cv.cycle_num, copy.deepcopy(OP_list)))
    
    
    fig = plt.figure(figsize = (4,3))
    ax = fig. add_axes([0.2,0.2,0.7,0.7])
    
    ax.set_xlabel('$j$ [mA/cm$^2$]')
    ax.set_ylabel('Onset potential [V vs SHE]')
    ax.set_xscale('log')
    ax.set_xticks([0.3, 0.5, 1, 2, 3])
    ax.set_xticklabels([0.3, 0.5, 1, 2, 3 ])
    

    for i,scan_data in enumerate(OP_each_scan):
        df_OP = pd.DataFrame(
            np.array(
                scan_data.OP
            ),
            columns=['log_j', 'onset potential']
        )
        y=df_OP['onset potential'].values
        x=10**df_OP['log_j'].values
        color = color_map_RGB(i/len(OP_each_scan))
        print(y)
        ax.scatter(x, y, color=color)
    
    ax.text(0.5, 0.815,"1st", c="#FF3333")
    ax.text(0.5, 0.93, "60th", c="#3333FF")


    plt.savefig("EC_onsetpotential.png", dpi=600)

CVs()


# %%
