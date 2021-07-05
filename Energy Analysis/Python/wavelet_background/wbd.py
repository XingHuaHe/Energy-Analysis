# 
# Description:
#       Iter wavelet background deduction.
#       迭代小波本底扣除.
# Author:
#       Xinghua.He
# History:
#       2020.10.26
# 
# 
# %%
import os
import datetime

import pywt
from pywt import wavedec, waverec, upcoef
import numpy as np
import matplotlib.pyplot as plt

def get_datas(filename):
    '''
    Description:
        Return energy.

    Input:
        filename    : file absolute path.
    '''

    energy = []
    with open(filename, 'rb') as f:
        data = f.read()
        m = 5
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            energy.append(temp)
            m += 3

    return energy

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

def wavelet_denoising(X, wavelet, mode, threshold):
    '''
    Description:
        Wavelet reconstruction according level.
    
    Input:
        X           : 输入.
        wavelet     : 选择采用的小波基.
        mode        : 采用的扩展模式.
        threshold   : 阈值.

    Return:
        .
    '''
    wave = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(X), wave.dec_len)
    coeffs = pywt.wavedec(X, wave.name, mode, maxlev)

    # 去噪
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    rc = pywt.waverec(coeffs, wave.name)

    return rc

def iter_wavelet_drop_background(X, wavelet, mode, theta, denoising, threshold):
    '''
    Description:
        Iter wavelet drop background.
    
    Input:
        X           : 输入.
        wavelet     : 选择采用的小波基.
        mode        : 采用的扩展模式.
        theta       : 迭代误差阈值.
        denoising   : 是否采用小波去噪.
        threshold   : 阈值.

    Return:
        background  : 本底.
    '''
    assert not denoising == None, 'denoising is not None.'
    assert not (denoising == True and threshold == None), 'threshold can\'t be None.'

    if denoising != None:
        # wavelet denosising.
        X = wavelet_denoising(X, 'db3', mode, threshold)

    # Iter wavelet drop background.
    count = 0
    times = 0
    background = X.copy()
    last_background = np.array(background)
    while True:
        # computed the max level.
        wave = pywt.Wavelet(wavelet)
        maxlev = pywt.dwt_max_level(len(background), wave.dec_len)
        
        # Wavelet decompese.
        coeffs = wavedec(background, wave.name, mode, level=maxlev)

        # Wavelet reconstruction.
        rc = wrcoef(coeffs, wavelet, mode, 0)

        # computed the background.
        for i in range(len(energy)):
            if background[i] > rc[i] and rc[i] > 0:
                background[i] = rc[i]
        
        # err = np.abs(last_background-np.array(background))[np.argmax(np.abs(last_background-np.array(background)))]
        # 相对误差
        err = np.sum(np.abs(last_background-np.array(background)) / last_background)
        if err < theta:
            times += 1
        else:
            times = 0

        last_background = np.array(background)

        if times >= 10:
            break

        count += 1
        if count > 200:
            break
    print(err)
    return background

# %
if __name__ == "__main__":
    # Initialization.
    wavelet_type = 'sym4'
    mode = 'symmetric'
    threshold = 0.05

    # Getting energy data.
    # rootpath = os.getcwd()
    rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    # filepath = os.path.join(rootpath, 'DatDatas', '数据-普通合金', '89094388914666093468345320200920111652291qe.dat')
    filepath = os.path.join(rootpath, 'DatDatas', 'wood.txt')
    energy = []
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            data = float(line.strip().strip('\n').strip('\t').split('\t')[-1].strip())
            energy.append(data)
    # energy = get_datas(filepath)
    # Select 1-2000 data point.
    energy = energy[0:1500]
    # %
    # Iter drop background.
    bg = iter_wavelet_drop_background(energy, wavelet_type, mode, 1e-3, True, threshold)

    # %
    # Drawing figures.
    _, ax = plt.subplots()
    ax.plot([i for i in range(len(energy))], energy, label='origin', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], bg, label='background', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], np.array(energy) - np.array(bg), label='drop_background', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{wavelet_type}')
    ax.legend()
    plt.show()

# %%
