# 
# Description:
#       Windows iter wavelet background deduction.
#       窗迭代自适应小波本底扣除.
# Author:
#       Xinghua.He
# History:
#       2020.11.03
# 
# 
# %%
import os
import datetime

import pywt
from pywt import wavedec, waverec, upcoef
import numpy as np
import matplotlib.pyplot as plt

# define wavelet to be selected.
wavelet_package = ['db5', 'db8', 'db30', 'sym4', 'bior6.8']

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
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 90
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

def iter_wavelet_drop_background(X, wavelet, mode, level=None):
    '''
    Description:
        Iter wavelet drop backgrounp.
    
    Input:
        coeffs  : 
        wavelet : 选择采用的小波基.
        mode    : 采用的扩展模式.
        level   : 利用第 level 层逼近系数进行重构.

    Return:
        background : reconstruction.
    '''
    # computed the max level.
    if level == None:
        wave = pywt.Wavelet(wavelet)
        maxlev = pywt.dwt_max_level(len(X), wave.dec_len)
    else:
        maxlev = level

    # 迭代去本底.
    count = 0
    theta = 1e-3
    times = 0
    background = X
    last_background = np.array(background)
    while True:
        # Wavelet decompese.
        coeffs = wavedec(background, wavelet, mode, level=maxlev)

        # Wavelet reconstruction.
        rc = wrcoef(coeffs, wavelet, mode, 0)

        # computed the background.
        for i in range(len(X)):
            if background[i] > rc[i] and rc[i] > 0:
                background[i] = rc[i]
        
        err = np.abs(last_background-np.array(background))[np.argmax(np.abs(last_background-np.array(background)))]
        # 相对误差
        # err = np.sum(np.abs(last_background-np.array(background)) / last_background)
        if err < theta:
            times += 1
        else:
            times = 0

        # last_rc = np.array(background)

        if times >= 5:
            break

        count += 1
        if count > 300:
            break

    return background

def wavelet_denoising(X, wavelet, mode, threshold):
    '''
    Description:
        Wavelet reconstruction according level.
    
    Input:
        X  : 输入.
        wavelet     : 选择采用的小波基.
        mode        : 采用的扩展模式.
        threshold   : 阈值.

    Return:
        
    '''
    wave = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(X), wave.dec_len)
    coeffs = pywt.wavedec(X, wave.name, mode, maxlev)

    # 去噪
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    rc = pywt.waverec(coeffs, wave.name)

    return rc

def adaptive_windows_iter_wavelet(X, wavelets, mode, windows, level=None):
    '''
    Description:
        Iter wavelet drop backgrounp.
    
    Input:
        X       : Input. 
        wavelet : 选择采用的小波基.
        mode    : 采用的扩展模式.
        windows : 窗个数.
        level   : 利用第 level 层逼近系数进行重构.

    Return:
        background : reconstruction.
    '''
    assert type(wavelets) == list, 'type(wavelets) is not list.'
    assert type(windows) == int or windows >=2, 'type(windows) is not int and windows size >= 2.'

    # split windows. sigbal window size.
    window_count = int(len(X) / windows)
    
    # computed wavelet reconstroction.
    reconstructs = []
    for i in range(len(wavelets)):
        temp = iter_wavelet_drop_background(X.copy(), wavelets[i], mode, level=None)
        reconstructs.append(temp)

    rc = np.zeros_like(X)
    c = 0
    index = [0, 0] # index of 0 represent best mean wavelet, and index 1 represent best var wavelet.
    best_average = 0
    best_var = 0
    for i in range(windows):
        for j in range(len(wavelets)):
            if i != (windows-1):
                # 计算均值
                average = np.mean(reconstructs[j][c:c+window_count])
                # 计算方差
                var = np.var(reconstructs[j][c:c+window_count])
            else:
                # 计算均值
                average = np.sum(reconstructs[j][c:])
                # 计算方差
                var = np.var(reconstructs[j][c:])

            if j == 0:
                index = [0, 0]
                best_average = average
                best_var = var
            else:
                if best_average < average:
                    best_average = average
                    index[0] = j
                if best_var > var:
                    best_var = var
                    index[1] = j

        # 判断均值与方差是否来自同一个小波分解.
        if index[0] == index[1]:
            if i != (windows-1):
                rc[c:c:c+window_count] = reconstructs[index[0]][c:c:c+window_count]
            else:
                rc[c:] = reconstructs[index[0]][c:]
        else:
            if i != (windows-1):
                rc[c:c+window_count] = (reconstructs[index[0]][c:c+window_count] + reconstructs[index[1]][c:c+window_count]) / 2
            else:
                rc[c:] = (reconstructs[index[0]][c:] + reconstructs[index[1]][c:]) / 2
            
        c = c + window_count

    return (rc, reconstructs)

def background_deduction(X):
    # 小波去噪.
    threshold = 0.08
    energy_denosising = wavelet_denoising(X, 'db8', 'symmetric', threshold)
    background = energy_denosising.copy()
    # 计算本底.
    s1, _ = adaptive_windows_iter_wavelet(background, wavelet_package, 'symmetric', 7, level=None)

    return (np.array(energy_denosising) - np.array(s1))

# %
if __name__ == "__main__":
    # Getting energy data.
    rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    # rootpath = os.path.abspath(os.path.dirname("__file__"))
    # filepath = os.path.join(rootpath, 'DatDatas', '数据-普通合金', '89094388914666093468345320200920111652291qe.dat')
    # energy = get_datas(filepath)
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

    # computed the max level.
    # wave = pywt.Wavelet('sym20')
    # maxlev = pywt.dwt_max_level(len(background), wave.dec_len)

    # 小波去噪.
    threshold = 0.04
    energy_denosising = wavelet_denoising(energy, 'db8', 'symmetric', threshold)
    background = energy_denosising.copy()

    s1, s2 = adaptive_windows_iter_wavelet(background, wavelet_package, 'symmetric', 7, level=None)
    # background = np.concatenate((background1, background2, background3, background4, background5))

    _, ax = plt.subplots()
    ax.plot([i for i in range(len(energy))], energy_denosising, label='origin', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], s1, label='mix', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], s2[0], label='s0', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], s2[1], label='s1', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], s2[2], label='s2', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], s2[3], label='s3', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], s2[4], label='s4', linewidth=0.5)
    ax.plot([i for i in range(len(energy))], np.array(energy_denosising) - np.array(s1), label='drop_backgrounp', linewidth=0.5)
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title('simple plot')
    ax.legend()
    plt.show()

# %%
