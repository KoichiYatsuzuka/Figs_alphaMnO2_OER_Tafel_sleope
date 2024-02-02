"""
図のファイル名がOth_から始まるもの(Othersの意)
トランペットプロットとそれに関連するものが主
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from collections import namedtuple
from dataclasses import dataclass
import sklearn.metrics as metrics

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.__str__()+'../../../')
# 共通部分の定義読み込み
import Common_for_figs
from Common_for_figs import F, R, T, SSCE_TO_SHE
from Common_for_figs import Point
from Common_for_figs import RedoxKinetics
Common_for_figs.set_common_matlptlib_rcParameter()
os.chdir(Path(__file__).resolve().parent.__str__())

keys_scan_num = [1, 9, 60] # 最初、いい感じに途中、最後のtrumept plot

#alpha_eff of OER vs EC features of pre-redox peak
def fit_trumpet(
    Ea: np.ndarray,
    Ec: np.ndarray,
    logv: np.ndarray,
    #mask_threshold = 0.1
    mask_point_from_last = 3
    )->tuple[RedoxKinetics, Point, Point, Point]:
    """ 
    単位変換などは行わない。
    この関数にそのような役割は持たせない。
    ## 引数:
        Ea: anode掃引時のピーク[V]のndarray
        Ec: cathode掃引時のピーク[V]のndarray
        logv: 掃引速度(V/s)の常用対数
        mask_point_from_last: 最高掃引速度から何plotを解析するか
    ## 返り値: 
        (alpha,n,k,E0), (x0,y0), (x1,y1), (x2,y2)の順のtuple
        alpha: 0から1の値
        n: 電子移動数
        k: 1/sに換算
        E0: V 参照極は変更しない
        (x0, y0): anode側fittingとcathode側fittingの交点
        (x1, y1): anode側のfittingの端
        (x2, y2): cathode側のfittingの端

    """
    #最後から該当する点だけ抽出
    peak_pos_anod:np.ndarray = Ea[-1*mask_point_from_last:]
    peak_pos_cath:np.ndarray = Ec[-1*mask_point_from_last:]
    ln_v:np.ndarray = (logv[-1*mask_point_from_last:])*np.log(10) 

    # 直線フィッティング
    fit1 = linregress(ln_v,peak_pos_anod)
    slope_anod,intercept_anod = fit1.slope, fit1.intercept 
    fit2 = linregress(ln_v,peak_pos_cath)
    slope_cath,intercept_cath = fit2.slope, fit2.intercept

    #交点とかを数学的に計算
    cross_x = (intercept_cath-intercept_anod)/(slope_anod-slope_cath)/np.log(10)
    cross_y = (slope_anod*intercept_cath-slope_cath*intercept_anod)/(slope_anod-slope_cath)
    cross_point = Point(cross_x, cross_y)
    
    x1 = ln_v[-1]/np.log(10)
    y1 = slope_anod*ln_v[-1]+intercept_anod
    edge_point_anod = Point(x1, y1)

    x2 = ln_v[-1]/np.log(10)
    y2 = slope_cath*ln_v[-1]+intercept_cath
    edge_point_cath = Point(x2, y2)

    # 物理化学パラメータの計算
    alpha = slope_cath/(slope_cath-slope_anod)
    A = slope_anod*alpha # this should be equal to slope_cath*(alpha-1)
    n = 1/(A*F/R/T)

    lnk = \
        alpha*(alpha-1)*(intercept_anod-intercept_cath)/A \
        +(alpha-1)*np.log(slope_anod)\
        -alpha*np.log(-slope_cath)
    k = np.exp(lnk) #

    # 最低掃引速度のピーク位置平均値から計算
    E0 = (Ea[0]+Ec[0])/2

    reaction_kinetics = RedoxKinetics(alpha, n, k, E0)

    return reaction_kinetics, cross_point, edge_point_anod, edge_point_cath

@dataclass
class PeakPosScanrateDependency:
    scan_rate: np.ndarray
    """unit: [V/s]"""
    anodic_peaks: np.ndarray
    cathodic_peaks: np.ndarray

@dataclass
class TrumpetResult:
    kinetic_params: RedoxKinetics
    cross_point: Point
    anodic_edge_point: Point
    cathodic_edge_point: Point

@dataclass
class TrumpetResEachScan:
    dict_trumpet_result: dict[int, TrumpetResult]

    def __getitem__(self, scan_num: int):
        return self.dict_trumpet_result[scan_num]

    @property
    def scan_numbers(self)->np.ndarray:
        return np.array(list(self.dict_trumpet_result.keys()))
    @property
    def alpha_ary(self)->np.ndarray:
        alpha_vals = []
        for key in self.dict_trumpet_result.keys():
            alpha_vals.append(self.dict_trumpet_result[key].kinetic_params.alpha)
        return np.array(alpha_vals)
    
    @property
    def n_ary(self)->np.ndarray:
        n_vals = []
        for key in self.dict_trumpet_result.keys():
            n_vals.append(self.dict_trumpet_result[key].kinetic_params.electron_n)
        return np.array(n_vals)
    
    @property
    def k0_ary(self)->np.ndarray:
        k0_vals = []
        for key in self.dict_trumpet_result.keys():
            k0_vals.append(self.dict_trumpet_result[key].kinetic_params.k0)
        return np.array(k0_vals)

    @property
    def E0_ary(self)->np.ndarray:
        E0_vals = []
        for key in self.dict_trumpet_result.keys():
            E0_vals.append(self.dict_trumpet_result[key].kinetic_params.E0)
        return np.array(E0_vals)
    
    def k_ary(self, applied_potential: float)->np.ndarray:
        E = applied_potential
        return copy.copy(
            self.k0_ary * np.exp(
                self.alpha_ary * F/R/T * (E-self.E0_ary)
            )
        )

def trumpet_analysis()->\
    tuple[ dict[int,PeakPosScanrateDependency], dict[int, TrumpetResult] ]:
    # ピークリストをファイルから取得
    for i, file in enumerate(["Trumpet_Anodic_Peaks.csv","Trumpet_Cathodic_Peaks.csv"]):
        
        # 左から三列を削除する
        # 最左列をindexカラム、最上段をカラム名となるように読み込む  
        tmp_all_column= pd.read_csv(file,index_col = 1)
        df = tmp_all_column.iloc[:,3:]

        # ファイルの構成を読み込む
        if i == 0:
            # 次元数取得、配列初期化
            L,N = df.shape
            peaks = np.zeros((2,L,N)) 

            # 共通となる情報の取得
            v = df.index.values/1000 #mV/sからV/sに変換
            log_v = np.log10(v)
            scan_nums = df.columns.values.astype(int)

        peaks[i] = df.values

    #返り値用
    peak_pos_each_scan = {} #ループを抜けたらそのまま返す
    tmp_trumpet_results = {} #あとでTrumpetResEachScanに変換して返す

    # trumpet plot解析開始
    for i,scan_num in enumerate(scan_nums):
        if scan_num>60: #CVスキャン数でいう60回目が終わった時点で停止
            break
        Ea,Ec = peaks[:,:,i]

        PeakList = PeakPosScanrateDependency(
            df.index.values/1000,
            Ea,
            Ec
        )
        peak_pos_each_scan[scan_num] = copy.copy(PeakList)
        
        tmp1, tmp2, tmp3, tmp4 = fit_trumpet(Ea,Ec,log_v)
        trumpet_result = TrumpetResult(tmp1, tmp2, tmp3, tmp4)

        tmp_trumpet_results[scan_num] = copy.copy(trumpet_result)

    trumpet_result_each_scan = TrumpetResEachScan(copy.copy(tmp_trumpet_results))

    #値の変更はここまでで完了

    """# ファイル出力
    output_txt = "index,alpha,n,k,E0\n"
    with open("Redox_parameters_from_trumpet.csv", "w") as f:
        results = trumpet_result_each_scan
        for scan_num in results.scan_numbers:
            kinetic_params = results.dict_trumpet_result[scan_num].kinetic_params
            alpha=kinetic_params.alpha
            n=kinetic_params.electron_n
            k0 = kinetic_params.k0
            E0 = kinetic_params.E0
            output_txt += f"{scan_num},{alpha},{n},{k0},{E0+0.2}\n"

        f.write(output_txt)"""

    return (peak_pos_each_scan, trumpet_result_each_scan)

# peak positionsの掃引速度依存性のplotとfitting
def draw_trumpet_plots(peak_pos_each_scan: dict[int,PeakPosScanrateDependency]):
    #描画キャンバス設定
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    c_list =['#FF3030','#308030','#3030FF','k']

    ax.set_xlabel(r'log$_{10}$ ($\nu$) [V/s]')
    ax.set_ylabel('$E$ [V vs. SHE]')

    for i in range(3):

        key = keys_scan_num[i]
        Ea: np.ndarray = peak_pos_each_scan[key].anodic_peaks + SSCE_TO_SHE
        Ec: np.ndarray = peak_pos_each_scan[key].cathodic_peaks+ SSCE_TO_SHE
        log_v = np.log10(peak_pos_each_scan[key].scan_rate)
        
        ax.scatter(log_v, Ea, c = c_list[i], label = key)
        ax.scatter(log_v, Ec, c = c_list[i])

        kinetics, cross_point, anodic_ege, cathodic_edge = fit_trumpet(Ea, Ec,log_v)
        ax.plot([cross_point.x, anodic_ege.x], [cross_point.y,anodic_ege.y],
            c=c_list[i], lw = 1)
        ax.plot([cross_point.x, cathodic_edge.x], [cross_point.y,cathodic_edge.y], 
            c=c_list[i], lw = 1)
        ax.axhline(kinetics.E0, c = c_list[i], ls = '--', lw =1)

    plt.savefig("Oth_Trumpet_Fitting.png", dpi=600)
    
    return

def draw_kinetic_parameters_changes(kinetic_params_each_scan: TrumpetResEachScan):
    fig = plt.figure(figsize = (4,6))
    c_list =['#FF3030','#308030','#3030FF','k']
    plt.subplots_adjust(left=0.2, right=0.90, top=0.98, bottom=0.08, hspace=0)

    right_y_labels = [
        r"$\alpha_1$ [-]",
        "$n_1$ [-]",
        "$k_1^0$ [1/s]",
        "$E_1$ [V]"
        ]
    kinetic_params = kinetic_params_each_scan
    right_y_series = [
        kinetic_params.alpha_ary,
        kinetic_params.n_ary,
        kinetic_params.k0_ary,
        kinetic_params.E0_ary+SSCE_TO_SHE
    ]
    # right_y_lims = [
    #   (0.4,0.7),log10_v
    #   (0.8,1.2),
    #   (1,4),
    #   (0.1,0.15)
    #   ]

    ax_right = []
    for i_kp in range(4): 
        # 各々のプロット領域準備
        # i_kp  0,      1,  2,  3
        # kp    alpha,  n,  k0, E0
        ax_right.append(fig.add_subplot(4,1,i_kp+1))
        ax_right[i_kp].set_ylabel(right_y_labels[i_kp])
        # ax_right.set_ylim(right_y_lims[i_kp])
        
        #上三つはx軸ラベルなし
        if i_kp != 3:
            ax_right[i_kp].set_xticklabels([])
        else:
            # 一番下にだけ共通のX軸ラベル
            ax_right[i_kp].set_xlabel("CV cycles")
        
        # プロット
        ax_right[i_kp].plot(kinetic_params.scan_numbers, right_y_series[i_kp], "k-o", lw=1.0)

        # annotation
        ## 左側でpick upされているtrumpet plotを示すための縦線
        for i_sn, sn in enumerate(keys_scan_num): #scan_num
            ax_right[i_kp].axvline(sn, linestyle="--", c=c_list[i_sn], lw=1)
    
    plt.savefig("Oth_kinetic_parametr_changes.png", dpi = 600)

    return

def draw_K_change(kinetic_params_each_scan: TrumpetResEachScan):
    
    APPLIED_POTENTIAL = 0.9
    
    k1_dict = {}
    k1r_dict = {}
    K_dict = {}
    
    for scan in kinetic_params_each_scan.scan_numbers:
        tmp = kinetic_params_each_scan.dict_trumpet_result[scan]
        k = tmp.kinetic_params.forward_rate(APPLIED_POTENTIAL)
        kr = tmp.kinetic_params.backward_rate(APPLIED_POTENTIAL)
        k1_dict[scan] = np.log10(k)
        k1r_dict[scan] = np.log10(kr)
        K_dict[scan] = np.log10(k/kr)

    #描画キャンバス設定
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.15,0.2,0.7,0.7])
    ax_left_k = ax
    ax_right_kr = ax.twinx()
    ax.set_xlabel("Scan number")
    ax_left_k.set_ylabel("log$_{10}(k_1$)")
    ax_right_kr.set_ylabel("log$_{10}(k_1r$)")

    ax_left_k.plot(
        k1_dict.keys(),
        k1_dict.values(),
        "-o",
        markersize=5,
        c="#FF3333",
    )
    
    ax_right_kr.plot(
        k1r_dict.keys(),
        k1r_dict.values(),
        "-o",
        markersize=5,
        c="#3333FF",
    )

    plt.savefig("Oth_k1_k1r_change.png", dpi=600)

    #描画キャンバス設定
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel("Scan number")
    ax.set_ylabel("log$_{10}$($K_1$) at " + str(APPLIED_POTENTIAL) + " V")

    ax.plot(
        K_dict.keys(),
        K_dict.values(),
        "-o",
        markersize=5,
        c="k",
    )
    plt.savefig("Oth_K1_change.png", dpi=600)

def k1_vs_Tafel(trumpet_res: TrumpetResEachScan):

    # k1 at 0.9 V vs NHE
    k1_ary_0_9V = trumpet_res.k_ary(0.7)
    print(trumpet_res)
    log_k1 = np.log10(k1_ary_0_9V)

    # loading Tafel slope data
    df_Tafel_data = pd.read_csv("Voltammogram_alpha_eff_data.csv")

    # (V/dec -> mV/dec)
    min_Tafel_ary_raw = df_Tafel_data["minimum Tafel slope"].values * 1000
    Tafel_ary = min_Tafel_ary_raw[0:len(k1_ary_0_9V)]

    #回帰一次直線
    ##関数定義
    def linear_func(x, slope, interrupt):
        return slope*x +interrupt
    popt, _ = curve_fit(linear_func, log_k1, Tafel_ary)
    fitting_line_y = linear_func(log_k1, *popt)
    r2 = metrics.r2_score(Tafel_ary, linear_func(log_k1, *popt))
    print(popt)
    print(r2)

    #canvas setting
    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel("log$_{10}$($k_1$ at 0.9 V) [s$^{-1}$]")
    ax.set_ylabel("Tafel slope [mV/dec]")
    

    tmp_x_edges = np.array(
        [log_k1[0], np.max(log_k1)], 
        dtype=float) 
    
    ax.plot(
        tmp_x_edges,
        linear_func(tmp_x_edges, *popt),
        c="#000000",
        linestyle = ":"
    )

    ax.scatter(
        log_k1,
        Tafel_ary,
        c = "#FF3333",
        s = 20
    )

    ax.text(
        5.5,70,
        "$R^2$ = {:.2f}".format(r2)
    )

    plt.savefig("Oth_k1_vs_Tafel.png", dpi=600)
    
    return 

def k_vs_scan(trumpet_res: TrumpetResEachScan):

    scans = trumpet_res.dict_trumpet_result.keys()

    k1_ary = trumpet_res.k_ary(0.7)

    fig = plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.set_xlabel("Scans")
    ax.set_ylabel("log ($k_1$ [1/s])")

    ax.scatter(
        scans,
        np.log10(k1_ary),
        s=12,
        c="r"
    )
    plt.savefig("k1_vs_scan.png")

    return

def trumpet():
    peak_pos, trumpet_res = trumpet_analysis()
    draw_trumpet_plots(peak_pos)
    draw_kinetic_parameters_changes(trumpet_res)
    #draw_trumpet_result(peak_pos, trumpet_res)
    #draw_K_change(trumpet_res)
    k1_vs_Tafel(trumpet_res=trumpet_res)
    k_vs_scan(trumpet_res)

trumpet()

# %%
