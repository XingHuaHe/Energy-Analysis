# 
# Description:
#       Wavelet background deduction and analysic.
#       测试小波python小波分解与重构.
# Author:
#       Xinghua.He
# History:
#       2020.11.02
# 
# 
# %%
import os
import datetime

import pywt
from pywt import wavedec, waverec, upcoef
import numpy as np
import matplotlib.pyplot as plt

def wrcoef(coeffs, wavelet, mode, level):
    '''
    Description:
        Wavelet reconstruction according level.
    
    Input:
        coeffs  : 重构系数矩阵.
        wavelet : 选择采用的小波基.
        mode    : 采用的扩展模式.
        level   : 利用第 level 层逼近系数进行重构.

    Return:
        wavelet reconstruction.
    '''
    assert not (len(coeffs)-1) < level, 'len(coeffs) is small than level.'

    for i in range(len(coeffs)):
        if i == (len(coeffs)-level) or i == 0:
            continue
        coeffs[i] = np.zeros_like(coeffs[i])

    return waverec(coeffs, wavelet, mode)

# %
if __name__ == "__main__":
    # Getting energy data.
    rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    filepath = os.path.join(rootpath, 'DatDatas', 'wood.txt')
    energy = []
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            data = float(line.strip().strip('\n').strip('\t').split('\t')[-1].strip())
            energy.append(data)
    energy = energy[0:1500]

    # 1
    coeffs1 = wavedec(energy.copy(), 'db5', 'symmetric', level=5)
    # Wavelet reconstruction.
    rc1 = wrcoef(coeffs1, 'db5', 'symmetric', 0)

    # 2
    coeffs2 = wavedec(energy.copy(), 'db5', 'symmetric', level=6)
    # Wavelet reconstruction.
    rc2 = wrcoef(coeffs2, 'db5', 'symmetric', 0)

    # 3
    coeffs3 = wavedec(energy.copy(), 'db5', 'symmetric', level=7)
    # Wavelet reconstruction.
    rc3 = wrcoef(coeffs3, 'db5', 'symmetric', 0)

    # 4
    coeffs4 = wavedec(energy.copy(), 'db5', 'symmetric', level=9)
    # Wavelet reconstruction.
    rc4 = wrcoef(coeffs4, 'db5', 'symmetric', 0)

    _, ax = plt.subplots()
    ax.plot([i for i in range(len(energy))], energy, label='origin', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], rc1, label='rc1', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], rc2, label='rc2', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], rc3, label='rc3', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], rc4, label='rc4', linewidth=0.5)
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title('simple plot')
    ax.legend()
    plt.show()
# %%
