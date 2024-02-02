"""
# Library for files from Nanophoton Raman microscope
"""

from . import common
import pandas as pd
import numpy as np
from typing import Optional

class RamanSpectrum:
	_data_name: str
	_wavenumber: np.ndarray[float]
	_intentsity: np.ndarray[float]

	def __init__(self, wavenumber, intensity, data_name):
		self._data_name = data_name
		self._wavenumber = wavenumber
		self._intentsity = intensity

	@property
	def data_name(self):
		return self._data_name
	
	@property
	def wavenumber(self):
		return self._wavenumber
	
	@property
	def intensity(self):
		return self._intentsity


def read_1D_data(file_path: str)->dict[str, RamanSpectrum]:


	"""extension = common.extract_extension(file_path)
	
	if extension != "txt":
		return None"""

	skip_line_num = \
		common.find_line_with_key(file_path, "#\n")+1
	
	df = pd.read_csv(file_path, sep="\t", skiprows=skip_line_num)

	column_names=df.keys()

	wavenumber_column_names = column_names[::2]
	spectrum_column_names = column_names[1::2]

	spectra: dict[str, RamanSpectrum] = {}
	for i in range(len(wavenumber_column_names)-1):
		spectra[spectrum_column_names[i]] = \
		RamanSpectrum(
			data_name = spectrum_column_names[i],
			wavenumber = df[wavenumber_column_names[i]].values,
			intensity= df[spectrum_column_names[i]].values
		)
		

	return spectra

