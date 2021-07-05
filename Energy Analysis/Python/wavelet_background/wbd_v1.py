# 
# Description:
#       Iter wavelet background deduction.
#       利用开源小波分解重构包进行小波本底扣除.
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
import numpy as np
import matplotlib.pyplot as plt
from waveletpb.wrcoef import wavedec, wrcoef

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
        denoising   : 是否采用小波去噪(bool). 
        threshold   : 阈值.

    Return:
        background  : 本底.
    '''
    assert not denoising == None, 'denoising is not None.'
    assert not (denoising == True and threshold == None), 'threshold can\'t be None.'

    if denoising == True:
        # wavelet denosising.
        X = wavelet_denoising(X, 'db5', mode, threshold)

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
        C, L = wavedec(background, wave, mode, maxlev)

        # Wavelet reconstruction.
        rc = wrcoef(C, L, wave, maxlev)

        # computed the background.
        for i in range(len(energy)):
            if background[i] > rc[i] and rc[i] > 0:
                background[i] = rc[i]
        
        err = np.abs(last_background-np.array(background))[np.argmax(np.abs(last_background-np.array(background)))]
        if err < theta:
            times += 1
        else:
            times = 0

        last_background = np.array(background)

        if times >= 3:
            break

        count += 1
        if count > 50:
            break
    
    return background

# %%
if __name__ == "__main__":
    # Initialization.
    wavelet_type = 'db5'
    mode = 'symmetric'
    threshold = 0.05

    # Getting energy data.
    # rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    # filepath = os.path.join(rootpath, 'DatDatas', 'wood.txt')
    filepath = os.path.join('D:\进行中的程序开发\Energy Classification\DatDatas', 'wood.txt')
    energy = []
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            data = float(line.strip().strip('\n').strip('\t').split('\t')[-1].strip())
            energy.append(data)

    bg = iter_wavelet_drop_background(energy, wavelet_type, mode, 10, False, None)

    _, ax = plt.subplots()
    ax.plot([i for i in range(2048)], energy, label='origin', linewidth=0.5)
    ax.plot([i for i in range(2048)], bg, label='background', linewidth=0.5)
    ax.plot([i for i in range(2048)], np.array(energy) - np.array(bg), label='drop_background', linewidth=0.5)
    ax.set_xlabel('x label')
    ax.legend()
    plt.show()

# %%
