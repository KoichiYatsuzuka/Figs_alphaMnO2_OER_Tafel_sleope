"""
### color_map_XtoY()
Expects a real number in the range of 0.0-1.0.
Returns tuple[float, float, float].

### color_array
The type is list[str]. 
Some major colors in hexdiciaml (i.g. #0088FF) are in this list.
"""


import math

#---------------------colors-----------------------------------------
def normalized_value_to_0to1(value: float):
    """
    三角派関数（になるような計算）を使って任意の実数を0~1.0の数値にする
    """
    return math.asin(math.sin(math.pi*(value-0.5)))/math.pi+0.5

def color_map_RGB(value: float):
    """
    0.0: red
    0.5: green
    1.0: blue
    """
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
    """
    0.0: black
    1.0: red
    """
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
    "#000000", # black
    "#888833", # dark yellow
    "#883388", # purple
    "#338888", # cyan
]
"""
Non-vivid colors.

"""