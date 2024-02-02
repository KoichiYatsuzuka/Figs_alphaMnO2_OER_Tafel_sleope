"""
CVの
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

DATA_FILE_NAME_BASE_1 = \
	"workspace_calcculate_alpha_eff/20221101_1-115_002_TrumpetPlot_ParaChange_18_CV_C01_{}.csv"

DATA_FILE_NAME_BASE_2 = \
	"workspace_calcculate_alpha_eff/20221101_1-115_003_TrumpetPlot_ParaChange_from10_every3_16_CV_C01_{}.csv"

WINDOW_LENGTH: int = 51

def load_alpha_and_Tafel(data_frame: pd.DataFrame)->tuple[list, list]:
	"""
	返り値はalpha, Tafelのlist
	ファイルの読み込み行数はリテラル直打ち（0.55 V - 0.845 Vになるはず）
	"""
	#data_frame=pd.read_csv(file_name)
	column_length = len(data_frame['E-IR/V'])
	alpha_eff_raw_values: list[float] =[]
	Tafel_raw_values: list[float] = []
	for index in range(0,column_length):
		if index<6 or (column_length-index)<6:
			# alpha_effとTafel slopeは移動平均を取っているため、
			# 最初と最後の5セルを無視
			continue
		
		if index < 1105 or index > 1700:
			# 0.55 V vs SSCE~ 0.845 V までのanodic scanではないなら無視
			continue

		#どちらにも該当しないときに値取得
		alpha_eff_raw_values.append(float(data_frame['alpha_eff'][index]))
		Tafel_raw_values.append(float(data_frame['Tafel slope mV/dec'][index]))
	return (alpha_eff_raw_values, Tafel_raw_values)

def integration_redox_charge(data_frame: pd.DataFrame, scan_rate: float = 5.0)->float:
	""" 0.05~0.2Vの間のanodic scanを積分して総クーロン量を返す\n
		data_frame: pd.read_csvで読み込んだ値\n
		scan_rate: 電位差→時間にするために使用する\n
		返り値: redox peakの電荷量 [mC]
	"""
	column_length = len(data_frame['E-IR/V'])
	#baselineの基準となる電位
	potential_baseline_begin:float = 0.05
	potential_baseline_end:float = 0.18
	#baseline基準を取得するindex(初期化はあとでしている)
	index_baseline_begin: int
	index_baseline_end: int
	#カラムの別名
	obsd_potential:np.ndarray = data_frame['Ewe/V'].values
	current_density:np.ndarray = data_frame['<I>/mA'].values

	
	for i in range(column_length):
		if obsd_potential[i] > potential_baseline_begin:
			index_baseline_begin = i
			break
	# iの値保持
	for j in range(i, column_length): 
		if obsd_potential[j]> potential_baseline_end:
			index_baseline_end = j
			break
	
	# 数学的にbaselineを一次関数として取得 i = slope*E + intercept
	baseline_slope:float = \
		(current_density[index_baseline_end]-current_density[index_baseline_begin])/\
		(obsd_potential[index_baseline_end]-current_density[index_baseline_begin])
	baseline_intercept:float = \
		current_density[index_baseline_end] - baseline_slope*obsd_potential[index_baseline_end]
	#print(baseline_slope, baseline_intercept)
	def delta_current(index: int)->float:
		return current_density[index] - baseline_slope*obsd_potential[index] + baseline_intercept
	
	total_charge: float = 0.0
	for i in range(index_baseline_begin, index_baseline_end):
		delta_time= (obsd_potential[i+1] - obsd_potential[i])/(scan_rate/1000)
		total_charge += delta_current(i) * delta_time

	return total_charge

def main():
	# keyはファイル名
	cycle_num: dict = {}
	alpha_eff_raw_curves: dict= {}
	alpha_eff_smoothed_curves: dict= {}
	alpha_eff_max_values: dict ={}
	Tafel_raw_curves:dict= {}
	Tafel_smoothed_curves:dict= {}
	Tafel_min_values={}
	redox_charge: dict = {}

	def common_procedure(file_name: str, _cycle_num: int):

		cycle_num[file_name] = _cycle_num

		data_frame = pd.read_csv(file_name)

		alpha_eff_raw_curves[file_name], Tafel_raw_curves[file_name]=load_alpha_and_Tafel(data_frame)
		
		# なんとかフィルター
		alpha_eff_smoothed_curves[file_name]=\
			savgol_filter(alpha_eff_raw_curves[file_name], window_length=WINDOW_LENGTH, polyorder = 2)

		Tafel_smoothed_curves[file_name]=\
			savgol_filter(Tafel_raw_curves[file_name], window_length=WINDOW_LENGTH, polyorder = 2)

		alpha_eff_max_values[file_name]=max(alpha_eff_smoothed_curves[file_name])
		Tafel_min_values[file_name]=min(Tafel_smoothed_curves[file_name])
		
		redox_charge[file_name] = integration_redox_charge(data_frame)

		return

	#10回目までのCV
	for file_num in range(0, 7):
		file_name=DATA_FILE_NAME_BASE_1.format(file_num)

		common_procedure(file_name, file_num+1)
		
		cycle_num[file_name]=file_num+1
		"""
		alpha_eff_raw_curves[file_name], Tafel_raw_curves[file_name]=load_alpha_and_Tafel(file_name)
		
		# なんとかフィルター
		alpha_eff_smoothed_curves[file_name]=\
			savgol_filter(alpha_eff_raw_curves[file_name], window_length=WINDOW_LENGTH, polyorder = 2)

		Tafel_smoothed_curves[file_name]=\
			savgol_filter(Tafel_raw_curves[file_name], window_length=WINDOW_LENGTH, polyorder = 2)

		alpha_eff_max_values[file_name]=max(alpha_eff_smoothed_curves[file_name])
		Tafel_min_values[file_name]=min(Tafel_smoothed_curves[file_name])"""

		# テスト描画
		"""fig = plt.figure(figsize = (4,3))
		ax = fig. add_axes([0.1,0.2,0.8,0.7])

		ax.set_xlabel(r'x')
		ax.set_ylabel('alpha')

		ax.plot(range(0,len(alpha_eff_raw_curves[file_name])), alpha_eff_raw_curves[file_name], 'k--')
		ax.plot(
			range(0,len(alpha_eff_smoothed_curves[file_name])),
			alpha_eff_smoothed_curves[file_name],
			'r'
			)"""

	
	# 10回目以降
	for i in range(0,35):
		file_num:int = i*3 #ファイル上のサイクル数
		file_name=DATA_FILE_NAME_BASE_2.format(file_num)

		common_procedure(file_name, 9+3*i)
	# テスト描画
		"""fig = plt.figure(figsize = (4,3))
		ax = fig. add_axes([0.1,0.2,0.8,0.7])

		ax.set_xlabel(r'x')
		ax.set_ylabel('alpha')

		ax.plot(range(0,len(Tafel_raw_curves[file_name])), Tafel_raw_curves[file_name], 'k--')
		ax.plot(
			range(0,len(Tafel_smoothed_curves[file_name])),
			Tafel_smoothed_curves[file_name],
			'r'
			)
"""
	# ファイル書き出し開始
	file_output = open("Voltammogram_alpha_eff_data.csv", 'w', encoding = "UTF-8" )
	file_output.write("cycle,file name,max alpha_eff,minimum Tafel slope,charge[mC],normalized charge\n")
	for file_name in alpha_eff_raw_curves.keys():
		file_output.write("{},{},{},{},{},{}\n".format(
			cycle_num[file_name],
			file_name,
			alpha_eff_max_values[file_name],
			Tafel_min_values[file_name],
			redox_charge[file_name],
			redox_charge[file_name]/float(max(redox_charge.values()))
		))
	file_output.close()

if __name__ == "__main__":
	main()





