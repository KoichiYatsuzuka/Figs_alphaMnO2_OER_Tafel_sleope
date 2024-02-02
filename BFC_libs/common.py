"""
### BFC_libs.common
"""
from __future__ import annotations
import numpy as np
from typing import Any, Literal, Optional
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from functools import wraps
import inspect
from typing import Union, NewType
from typing import Generic, TypeVar, Final
from dataclasses import dataclass
import abc
from numpy import ufunc

from numpy._typing import NDArray


#------------------------------------------------------
#------------------decorators--------------------------
#------------------------------------------------------

def immutator(func):
	"""
	This decorator passes deepcopied aruments list.
	It is guaranteeed that the all original arguments will not be overwritten.
	"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		args_copy = tuple(copy(arg) for arg in args)
		kwargs_copy = {
			key: copy(value) for key, value in kwargs.items()
		}
		return func(*args_copy, **kwargs_copy)
	return wrapper

def self_mutator(func):
	"""
	This decorator passes deepcopied aruments list other than itself.
	It is guaranteeed that the all original arguments will not be overwritten.
	"""
	"""
		TO DO: 可能であれば、第一引数がselfかどうかのチェックをしたい。
		現在の問題点として、第一引数のオブジェクトが有するメソッドど同名のグローバル関数にこのデコレータを付けても問題なく動いてしまう。
	"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		"""
		"""

		# Checking wheatehr the first arg is self
		
		# Check the length of the arguments and get the firstr argument
		if len(args)==1:
			first_arg = args
		elif len(args)==0:
			raise InvalidDecorator("This decorator is used in class method with a self argument.")
		else:
			first_arg = args[0]
		
		# extract the name of the method which is calling this decorator
		func_name = func.__name__


		# try to get the id of class method
		try:
			first_arg.__getattribute__(func_name)

		except AttributeError:
			# Meaning that the object does not have the method whose name is tha same as the method calling this decorator.
			raise InvalidDecorator("This decorator must be used in class method.")
		
		"""
		
		func_id = id(non_wrapped_func)

		if func_id!=first_args_method_id:
			# Meaning that the most evil thing:
			# The object has the method whose name is the same as the method calling this decorator, but not class method.
			print(func)
			print(first_args_method)
			raise InvalidDecorator("This decorator must be used in class method.")"""
	
		if len(args)==1:
			args_copy = args
		else:
			args_copy = tuple([args[0]]) + \
				tuple(copy(arg) for arg in args[1:])
		kwargs_copy = {
			key: copy(value) for key, value in kwargs.items()
		}
		return func(*args_copy, **kwargs_copy)
	return wrapper
"""
def value_object(cls: type):
	
	# wrqp関数
	def wrap(cls):
		# 関数の実態定義
		def innitialization(self: cls, value: float):
			if type(value)!=float:
				error_report_raw = "An invalid object was substituted.\nAllowd type is float, but "+str(type(value))+" was used"
				error_report_blacket_replaced = error_report_raw.replace("<", "").replace(">", "")
				raise TypeError(error_report_blacket_replaced)
			self._value = value
		
		@property
		def value(self: cls):
			return self._value
		
		@immutator
		def addition (self: cls, added_value: cls)->cls:
			if type(self)!=type(added_value):
				error_report_raw = "An invalid object was substituted.\nAllowd type is "+str(type(self))+", but "+str(type(added_value))+" was used"
				error_report_blacket_replaced = error_report_raw.replace("<", "").replace(">", "")
				raise TypeError(error_report_blacket_replaced)
		
			return cls(self.value + added_value.value)
		
		@immutator
		def cast_to_float(self: cls)->float:
			return self.value
		
		@immutator
		def cast_to_str(self: cls)->str:
			return str(self.value)
		
		@immutator
		def print(self: cls)->str:
			return str(self.value)
		# 関数オブジェクトの代入
		cls.__init__ = innitialization
		cls.value = value
		cls.__add__ = addition
		cls.__float__ = cast_to_float
		cls.__str__ = cast_to_str
		cls.__repr__ = print

		return cls
	return wrap"""


#------------------------------------------------------
#----------classes and relative variables--------------
#------------------------------------------------------

NOT_ALLOWED_ERROR_STR: Final[str] = "An invalid object was substituted.\nAllowd type is {}, but {} was used"
OPERATION_ALLOWED_TYPES: Final[list[type]] = [
	float,
	int,
	np.float32,
	np.float64
]

def typeerror_other_type(self, another):
	if type(another)!=type(self):
			error_report = \
				NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(another))).\
				replace("<", "").replace(">", "")
			raise TypeError(error_report)

class ValueObject:
	_value: float

	def __init__(self, value: float):

		if not(type(value) in OPERATION_ALLOWED_TYPES) and type(value) != type(self):
			error_report = NOT_ALLOWED_ERROR_STR.format(OPERATION_ALLOWED_TYPES, str(type(value))).replace("<", "").replace(">", "")
			raise TypeError(error_report)

		self._value = float(value)
		
	@property
	def value(self)->float:
		return self._value
	
	@immutator
	def __add__(self, added_value):
		# error
		if type(self)!=type(added_value):
			error_report = \
				NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(added_value))).\
				replace("<", "").replace(">", "")
			raise TypeError(error_report)
		
		# normal process
		cls_type=type(self)
		sum: cls_type = cls_type(self.value+added_value.value)
		return sum
	
	@immutator
	def __sub__(self, subed_value):
		# error
		if type(self)!=type(subed_value):
			error_report = \
				NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(subed_value))).\
				replace("<", "").replace(">", "")
			raise TypeError(error_report)
		
		# normal process
		cls_type=type(self)
		diff: cls_type = cls_type(self.value-subed_value.value)
		return diff
	
	@immutator
	def __mul__(self, muled_value: Union[int, float, ValueObject]):
		# error
		if not(type(muled_value) in OPERATION_ALLOWED_TYPES) and type(muled_value)!= type(self):
			error_report = \
				NOT_ALLOWED_ERROR_STR.format(str(OPERATION_ALLOWED_TYPES), str(type(muled_value))).\
				replace("<", "").replace(">", "")
			raise TypeError(error_report)
		
		# normal process
		cls_type=type(self)
		product: cls_type = cls_type(self.value*float(muled_value))
		return product
	
	@immutator
	def __truediv__(self, dived_value: Union[int, float, ValueObject]):
		# error
		if type(dived_value)!=int and type(dived_value)!= float and type(dived_value)!= type(self):
			error_report = \
				NOT_ALLOWED_ERROR_STR.format(str(Union[float, int, type(self)]), str(type(dived_value))).\
				replace("<", "").replace(">", "")
			raise TypeError(error_report)
		
		cls_type=type(self)
		quotient: cls_type = cls_type(self.value / float(dived_value))
		return quotient
	
	@immutator
	def __lt__(self, another):
		# error
		try:
			typeerror_other_type(self, another)
		except ValueError as error_report:
			raise ValueError(error_report)
		
		return self.value<another.value

	@immutator
	def __le__(self, another):
		# error
		try:
			typeerror_other_type(self, another)
		except ValueError as error_report:
			raise ValueError(error_report)
		
		return self.value<=another.value
	
	@immutator
	def __gt__(self, another):
		# error
		try:
			typeerror_other_type(self, another)
		except ValueError as error_report:
			raise ValueError(error_report)
		
		return self.value>another.value
	
	@immutator
	def __ge__(self, another):
		# error
		try:
			typeerror_other_type(self, another)
		except ValueError as error_report:
			raise ValueError(error_report)
		
		return self.value>=another.value
	


	@immutator
	def __str__(self):
		
		return str(self.value)
	
	@immutator
	def __repr__(self)->str:
		
		return str(self.value)
	
	@immutator
	def __float__(self):
		
		return self.value
	
	@immutator
	def __abs__(self):
		return np.abs(self._value)



class ValueObjectArray(np.ndarray):
	"""
	Succeeds numpy.ndarray
	additional method
		normalize()
		float_array()
		find()
	"""
	
	"""
	自分用メモ
	何かあればこのQiita
	https://qiita.com/Hanjin_Liu/items/02b9880d055390e11c8e
	"""

	def __new__(cls, obj, dtype, meta: Optional[str] = None):
		self = np.asarray(list(map(dtype, obj)), dtype=dtype).view(cls)
		
		match meta:
			case None:
				self.meta=""
			
			case _:
				self.meta = meta
		
		return self
	
	def __array_finalize__(self, obj: Optional[NDArray[Any]]):
		#おそらく動いていないが、必要になったら改変
		if obj is None:
			return None
		self.meta = getattr(obj, "meta", None)

	def __array_ufunc__(self, ufunc: ufunc, method, *args: Any, **kwargs: Any):
		metalist = [] # メタ情報のリスト
		args_ = [] # 入力引数のリスト
		for arg in args:
			# 可能ならメタ情報をリストに追加
			if isinstance(arg, self.__class__) and hasattr(arg, "meta"):
				metalist.append(arg.meta)
			# MetaArrayはndarrayに直す
			arg = arg.view(np.ndarray) if isinstance(arg, ValueObjectArray) else arg
			args_.append(arg)
		# 関数を呼び出す
		out_raw = getattr(ufunc, method)(*args_, **kwargs)
		
		# なんか必要らしい
		if out_raw is NotImplemented:
			return NotImplemented

		# 型を戻す。このとき、スカラー(np.float64など)は変化しない。
		out = out_raw.view(self.__class__) if isinstance(out_raw, np.ndarray) else out_raw

		# メタ情報を引き継ぐ。このとき、入力したメタ情報を連結する。
		if isinstance(out, self.__class__):
			#print(metalist)
			#print(ufunc.__name__)
			out.meta = ','.join(metalist)+"_"+ufunc.__name__

		return out

	@immutator
	def normalize(self):
		normd_value: ValueObjectArray = (self-self.min())/(self.max()-self.min())
		normd_value.meta = normd_value.meta.removesuffix("_subtract_divide")+"_normalized"
		return normd_value
	
	@immutator
	def float_array(self)->np.ndarray:
		return np.array(self, dtype=float)
	
	@immutator
	def find(self, target_value, begin_index: int = 0, end_index: int = None)->Optional[list[int]]:
		if end_index == None:
			_end_index = len(self)-1
		
		if begin_index < 0 or begin_index > _end_index or _end_index > len(self)-1:
			raise IndexError("Invalid index. begin: {}, end: {} length: {}".format(begin_index, _end_index, len(self)))
		
		tmp_ary = self[begin_index:_end_index]
		explored_ary = tmp_ary-type(self[0])(target_value)


		index_list = np.where(np.delete(explored_ary, [0])*np.delete(explored_ary, [-1]) < type(self[0])(0))[0]
		

		return index_list

		
		index_list = []
		for i in range(begin_index, _end_index-1):
			if (self[i].value - target_value)*(self[i+1].value - target_value)<0:
				index_list.append(i)
		
		if len(index_list)>0:
			return index_list
		else:
			return None

@dataclass(frozen=True)
class DataFile(metaclass=abc.ABCMeta):
	_comment: list[str]
	_condition: list[str]
	_file_path: str

	@property
	def condition(self):
		return self._condition
	
	@property
	def comment(self):
		return self._comment
	
	@property
	def file_name(self):
		return self._file_path
	
@dataclass(frozen=True)
class DataSeriese(metaclass=abc.ABCMeta):
	_comment: list[str]
	_condition: list[str]
	_original_file_path: str

	@property
	def condition(self):
		return self._condition

	@property
	def comment(self):
		return self._comment
	
	@property
	def original_file_path(self):
		return self._original_file_path
	
	@immutator
	def plot(self, fig: Optional[plt.Figure]=None, ax: Optional[plt.Axes]=None, **args)->tuple[plt.Figure, plt.Axes]:
		"""
		データの簡易プロット用クラスメソッド
		"""
		match fig:
			case None:
				_fig = plt.figure(figsize = (4,3))
			case _:
				_fig = fig
		
		match ax:
			case None:
				_ax = _fig.add_axes([0.2,0.2,0.7,0.7])
			
			case _:
				_ax = ax
		return (_fig, _ax)


#------------------------------------------------------
#---------------common object values-------------------
#------------------------------------------------------	
		
class Time(ValueObject):
	pass

Time_Array = NewType("Time_Array", ValueObjectArray[Time])

#------------------------------------------------------
#-------------------functions--------------------------
#------------------------------------------------------

"""@immutator
def to_value_object_array(list:list, type: type, meta: Optional[str] = None)->ValueObjectArray[type]:
	
	converted_array= np.array([])
	for value in list:
		converted_array = np.append(converted_array, type(value))
	
	return copy(ValueObjectArray(converted_array, dtype=type, meta=meta))"""


def set_matpltlib_rcParameters()->None:
	"""
	Set parameters list:
	"font.size" = 10
	'axes.linewidth' = 1.5
	"xtick.top" = True
	"xtick.bottom" = True
	"ytick.left" = True
	"ytick.right" = True
	'xtick.direction' = 'in'
	'ytick.direction' = 'in'
	"xtick.major.size" =6.0
	"ytick.major.size" = 6.0
	"xtick.major.width" = 1.5
	"ytick.major.width" = 1.5
	"xtick.minor.size" =4.0
	"ytick.minor.size" = 4.0
	"xtick.minor.width" = 1.5
	"ytick.minor.width" = 1.5
	plt.rc('legend', fontsize=7)
	'lines.markersize' =3\n
	Any others? Do by yourselves.
	## Returns
		nothing
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
	plt.rcParams["ytick.minor.width"] = 1.5
	plt.rc('legend', fontsize=7)
	plt.rcParams['lines.markersize'] =6
	return

def create_standard_matplt_canvas()->tuple[plt.Figure, plt.Axes]:
    """
    ## Returns
    Returns tuple of usual Figure and Axes instances.
	fig = plt.figure(figsize = (4,3)),\n
	ax = fig.add_axes([0.2,0.2,0.7,0.7])
	"""
    fig=plt.figure(figsize = (4,3))
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    return (fig, ax)

def extract_extension(file_path: str)->Optional[str]:
	"""
	### Extract the extension from a file name.
	ig. test.txt -> txt
		/data/data.mpt -> mpt
		spec_0.5V.spc -> spc
	## parameter
	file_path: 
		Targetted file name.
		Relative path or absolute path is also acceptable.
		This can contain period other than extension.

	## Return Value
	The extracted extension in str type.
	If the parameter does not inculde period, this function returns None.
		
	## Error
	TypeError
		The parameter accepts only str. Any other types cause TypeError.

	"""
	try:
		splitted_str = file_path.split(sep=".")
	except(AttributeError):
		raise(TypeError(
			"Invalid file path: Parameter is not str. \nThe type is {}.".format(type(file_path))))
	if len(splitted_str) <2:
		return None
	return splitted_str[len(splitted_str)-1]

def extract_filename(file_path: str)->str:
	splitted_str_slash = file_path.split(sep="/")
	splitted_str_backslash = splitted_str_slash[len(splitted_str_slash)-1].split(sep="\\")
	return splitted_str_backslash[len(splitted_str_backslash)-1]

def find_line_with_key(
		file_path: str, 
		key_word: str,
		)->Optional[int]:
	"""
	### Count the number of lines firstly include the key word.
	ig.\n
	file contents=
		date: 2001/2/3 <- skip\n
		method: CV <- skip\n
		time, potential, current <- column name \n
		0, 0, 0.1 <- data row \n
		1, 0.005, 0.2 \n
		.......
	key_word = "potential"\n
	In this case, this function will return 3. 
	Read as UTF-8. If file is written in Shift-JIS, it may cause an error.

	## Parameters
	file_path: the path to the file to read.
			Relative or absolute patha is acceptable.
	key_word: the word to find.

	## Return value
	The position of the line firstly including the key word
	None: the key word was not found.
	"""
	# Reading each line with finding the key word
	file = open(file_path, 'r', encoding='UTF-8')
	lines = file.readlines()

	i_line_count :int = 0
	for line in lines:
		if line.find(key_word) != -1:
			# now the key word was found. 
			# current i values is (the number of lines read) - 1
			break
		i_line_count += 1

	if i_line_count == len(lines):
		# the key word was not found
		return None

	# succesfully finished
	return i_line_count+1



#------------------------------------------------------
#------------------exceptions--------------------------
#------------------------------------------------------
class InvalidDecorator(Exception):
	"""
	If a decorator is not used as expected, this error must be raised. 
	"""
	pass



