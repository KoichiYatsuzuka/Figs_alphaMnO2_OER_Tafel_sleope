"""
複数のpyファイルで使う定数、型、関数定義
"""
#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from typing import TypeVar, Generic

def set_common_matlptlib_rcParameter():
    """
        matplotlibのグラフ描画に関するパラメーターのうち、軸の太さなど共通なものを設定する\n
        # この関数以外でrcParamsを変えない。
    """
    plt.rcParams["font.size"] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["xtick.top"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["xtick.major.size"] =6.0
    plt.rcParams["ytick.major.size"] = 6.0
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.minor.size"] =4.0
    plt.rcParams["ytick.minor.size"] = 4.0
    plt.rcParams["xtick.minor.width"] = 1.5
    plt.rcParams["ytick.minor.width"] = 2.0
    plt.rc('legend', fontsize=7)
    plt.rcParams['lines.markersize'] =3
    return

# CONST VALUES
F = 96485
R = 8.314
T = 298
ELECTRON_NUM_OER = 4
SSCE_TO_SHE = 0.2
#%%

#---------------------colors-----------------------------------------
def normalized_value_to_0to1(value: float):
    """
    三角派関数（になるような計算）を使って任意の実数を0~1.0の数値にする
    """
    return math.asin(math.sin(math.pi*(value-0.5)))/math.pi+0.5

def color_map_RGB(value: float):
    #数値を0~1の値に変換
    value_convd = normalized_value_to_0to1(value)
    
    if value_convd < 0.5:
        red = 255 - 150*value_convd*2
        green = 28 + 100*value_convd*2
        blue = 55
    else:
        red = 55
        green =  128 - 100*(value_convd-0.5)*2
        blue = 55 + 200*(value_convd-0.5)*2
    
    return (red/255, green/255, blue/255)

def color_map_KtoR(value: float):
    #数値を0~1の値に変換
    value_convd = normalized_value_to_0to1(value)

    red = 255 * value_convd
    green =  50 * value_convd
    blue = 50 * value_convd

    return (red/255, green/255, blue/255)

def color_map_RtoB(value: float):
    """
    始点 0.0:  (230, 28, 20) 赤
    中間 0.5:  (125, 28, 125) 紫
    終点 1.0:  (20, 28, 230) 青
    """
    value_convd = normalized_value_to_0to1(value)

    red = 230 - 210 * value_convd
    green = 28
    blue = 20 + 210*value_convd

    return (red/255, green/255, blue/255)

color_array = [
    "#FF3333", # red
    "#338833", # green
    "#3333FF", # blue
    "#888833", # dark yellow
    "#883388", # purple
    "#338888", # cyan
    "#000000", # black
]

#%%
# user-defined objects
Point = namedtuple('Point', 'x y')
"""描画に使う座標"""

#-----------------------for XRD------------------------------------
def plot_PDF(
        ax: plt.Axes, 
        dataframe_PDF: pd.DataFrame,
        y_offset: float, 
        y_lims: tuple[float, float], 
        y_scale: float = 0.1,
        color: str = "k"
    ):
    
    d_value = dataframe_PDF['d value'].values
    intensity = dataframe_PDF['intensity'].values

    y_offset_relative = (y_offset-y_lims[0])/(y_lims[1]-y_lims[0])

    #そのままだとなぜかズレるので、y下方向に5%ずらす
    ax.axhline(y_offset-(y_lims[1]-y_lims[0])*0.005, xmin=0, xmax=1, c="k", lw=0.5)
    
    for i in range(len(d_value)):
        ax.axvline(
            # 線源Kalpha1 -> Kalpha; d値から換算
            np.rad2deg(2*np.arcsin (1.5418/2/d_value[i])), 
            c=color, lw = 1, 
            ymin = y_offset_relative,
            # データ値座標→グラフの相対座標 
            ymax = y_offset_relative + intensity[i]*y_scale
            )


#----------------------simulation and trumpet plot-----------------------------------------------------------
@dataclass
class ScanNumber:
    """
    リテラル数値絶対殺すマン
    =====================================
    """
    scan_number: int
    
    def __int__(self):
        return self.scan_number

    def __eq__(self, others):
        if type(self)!=type(others):
            raise TypeError(
                "ScanNumber.__eq__() is called, "+\
                "but the type of right side is {}.\n".format(type(others)) + \
                "If you used int as index, "+ \
                "be sure that an instance uses ScanNumber as index instead of int."
            )
        return self.scan_number == others.scan_number


@dataclass
class RedoxKinetics:
    alpha: float
    electron_n: float
    k0: float
    E0: float
    
    def forward_rate_array(self, applied_E_array)->np.ndarray:
        delta_E = applied_E_array - self.E0
        numerator = self.alpha *  F * delta_E
        denominator = R*T
        
        return self.k0*np.exp(numerator/denominator)
    
    def forward_rate(self, applied_E)->float:
        delta_E = applied_E - self.E0
        numerator = self.alpha *  F * delta_E
        denominator = R*T
        return self.k0*np.exp(numerator/denominator)


    def backward_rate_array(self, applied_E_array)->np.ndarray:
        delta_E = applied_E_array - self.E0
        numerator = - (1-self.alpha) *  F * delta_E
        denominator = R*T
        return self.k0*np.exp(numerator/denominator)
    
    def backward_rate(self, applied_E)->float:
        delta_E = applied_E - self.E0
        numerator = - (1-self.alpha) *  F * delta_E
        denominator = R*T
        return self.k0*np.exp(numerator/denominator)

class RedoxKineticsArray:
    _scan_nums_: list[ScanNumber]
    _redox_kinetcs_array_: list[RedoxKinetics]

    def __init__(self):
        self._scan_nums_ = []
        self._redox_kinetcs_array_ = []
    
    def __init__(self, list_scan_nums, list_redox_kinetics):
        
        #不正な引数は赦さない

        #長さが違うならはじく
        if len(list_scan_nums)!=len(list_redox_kinetics):
            raise ValueError(
                "Both of the lists must be same length: \n"+\
                "list_scan_num is {}, \n".format(len(list_scan_nums))+\
                "list_redox_kinetics is {}.\n".format(len(list_redox_kinetics))
                )
        # 型が違うならはじく
        for scan_num in list_scan_nums:
            if type(scan_num)!=ScanNumber:
                raise TypeError("A non-ScanNumber object is in the first argument\n")
        for redox_kinetic in list_redox_kinetics:
            if type(redox_kinetic)!=RedoxKinetics:
                raise TypeError("A non-RodexKinetic object is in the second argument\n")
        
        self._scan_nums_ = copy.copy(list_scan_nums)
        self._redox_kinetcs_array_ = copy.copy(list_redox_kinetics)

    def __setitem__(self, new_scan_num, new_redox_kinetic):
        """
        データの追加。
        indexが同じ値なら上書き、そうでなければ追加する。
        """
        if type(new_redox_kinetic)!=RedoxKinetics:
            raise TypeError("Type of right side must be RedoxKinetics")

        if new_scan_num in self._scan_nums_:
            index = self._scan_nums_.index(new_scan_num)
            self._redox_kinetcs_array_[index] = copy.copy(new_redox_kinetic)
        else:
            self._scan_nums_.append(new_scan_num)
            self._redox_kinetcs_array_.append(copy.copy(new_redox_kinetic))

    def __getitem__(self, scan_number):
        try:
            index = self._scan_nums_.index(scan_number)
            return self._redox_kinetcs_array_[index]
        except(TypeError):
            raise TypeError("ScanNumber class must be used as a index, not integer")

    @property
    def scan_nums(self):
        return self._scan_nums_
    
    @property
    def alpha_values(self):
        alpha_values = []
        for params in self._redox_kinetcs_array_:
            alpha_values.append(params.alpha)
        return np.array(alpha_values)
    
    @property
    def electron_n_values(self):
        electron_n_values = []
        for params in self._redox_kinetcs_array_:
            electron_n_values.append(params.electron_n)
        return np.array(electron_n_values)
    
    @property
    def k0_values(self):
        k0_values = []
        for params in self._redox_kinetcs_array_:
            k0_values.append(params.k0)
        return np.array(k0_values)

    @property
    def E0_values(self):
        E0_values = []
        for params in self._redox_kinetcs_array_:
            E0_values.append(params.E0)
        return np.array(E0_values)

@dataclass
class SimdVoltammogram:
    potential: np.ndarray
    current: np.ndarray

    @property
    def log_current(self)->np.ndarray:
        return np.log10(np.abs(self.current))

    @property
    def alpha_eff(self):
        alpha_vals = [1E-14]
        for i in range(len(self.current)-1):
            delta_E = self.potential[i+1] - self.potential[i]
            delta_log_i = self.log_current[i+1] - self.log_current[i]
            alpha_vals.append(0.059 * delta_log_i/delta_E)
        return np.array(alpha_vals)
    
    @property
    def Tafel_slope(self):
        """mV/dec"""
        return 0.059/self.alpha_eff * 1000


class ExperimentalVoltammogram:
    #cycle_num: int = 0 #何回CVを掃引したときのデータか
    """recorded_potential: list =[] #ポテスタで記録される生の電位
    corrected_potential: list = [] #iR補正をした電位
    corrected_potential_anodic :list =[] #iR補正の値のうち、最初の半分（おのずとanodic sweep分）
    current_density_mA: list = [] #電流密度（mA/cm2）
    log_current_density: list = [] #log_10 (current [A/cm2])"""

    """def __init__(
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
        self.log_current_density = log_current_density"""
    
    def __init__(self, data_frame: pd.DataFrame):
        self._data_frame_ = data_frame
    
    @property
    def anodic_edge(self)->int:
        return int(len(self._data_frame_['Ewe/V'].values)/2)

    @property
    def recorded_potential(self)->np.ndarray:
        return self._data_frame_['Ewe/V'].values
    
    @property
    def corrected_potential(self)->np.ndarray:
        return self._data_frame_['E-IR/V'].values
    
    @property
    def current_mA(self)->np.ndarray:
        return self._data_frame_['<I>/mA'].values
    
    @property
    def log_current(self)->np.ndarray:
        return self._data_frame_['log(|I|)'].values
    
    @property
    def Tafel_slope(self)->np.ndarray:
        tmp = self._data_frame_['Tafel slope mV/dec'].values
        tmp2 = np.where(tmp == ' ', np.NaN, tmp)
        return tmp2.astype(dtype=float)*1000 #V/dec -> mV/dec
    
    @property
    def alpha_eff(self)->np.ndarray:
        tmp = self._data_frame_['alpha_eff'].values
        tmp2 = np.where(tmp == ' ', np.NaN, tmp)
        return tmp2.astype(dtype=float)

Ty = TypeVar('Ty')

class VoltammogramArray(Generic[Ty]):
    _scan_nums_: list[ScanNumber]
    _voltammograms_: Ty

    def __init__(self):
        self._scan_nums_ = []
        self._voltammograms_ = []

    def __init__(self, list_scan_nums, list_voltammogram):
        
        #不正な引数は赦さない

        #長さが違うならはじく
        if len(list_scan_nums)!=len(list_voltammogram):
            raise ValueError(
                "Both of the lists must be same length: \n"+\
                "list_scan_num is {}, \n".format(len(list_scan_nums))+\
                "list_voltammogram is {}.\n".format(len(list_voltammogram))
                )
        # 型が違うならはじく
        for scan_num in list_scan_nums:
            if type(scan_num)!=ScanNumber:
                raise TypeError("A non-ScanNumber object is in the first argument\n")
        """for voltammogram in list_voltammogram:
            if type(voltammogram)!=Ty:
                print("Argument is "+str(type(voltammogram)))
                raise TypeError("A non-suitable voltammogram object is in the second argument.")"""        
        self._scan_nums_ = copy.copy(list_scan_nums)
        self._voltammograms_ = copy.copy(list_voltammogram)
    
    def __setitem__(self, new_scan_num, new_voltammogram):
        """
        データの追加。
        indexが同じ値なら上書き、そうでなければ追加する。
        """
        if type(new_voltammogram)!=Ty:
            raise TypeError("Type of right side must be {}".format(Ty))

        if new_scan_num in self._scan_nums_:
            index = self._scan_nums_.index(copy.copy(new_scan_num))
            self._voltammograms_[index] = copy.copy(new_voltammogram)
        else:
            self._scan_nums_.append(copy.copy(new_scan_num))
            self._voltammograms_.append(copy.copy(new_voltammogram))

    def __getitem__(self, scan_number)->Ty:
        try:
            index = self._scan_nums_.index(scan_number)
            return self._voltammograms_[index]
        except(TypeError):
            raise TypeError("ScanNumber class must be used as a index, not integer")
        except(ValueError):
            raise ValueError("The scan number of {} is not in list".format(int(scan_number)))
    
    @property
    def scan_nums(self)->list[ScanNumber]:
        return self._scan_nums_

class ExpVoltammogramList(VoltammogramArray[ExperimentalVoltammogram]):
    pass

class SimdVoltammogramList(VoltammogramArray[SimdVoltammogram]):
    pass

