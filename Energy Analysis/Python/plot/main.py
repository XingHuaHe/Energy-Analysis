# %%
# System packages.
import os
import sys
import argparse

# Extend packages.
import numpy as np
import pandas as pd
import scipy.io as scio
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.rc('font', weight='bold') 

sns.set_theme(color_codes=True)
sns.set_style("whitegrid")
# Current directory path.
ROOTPATH = os.path.dirname(os.path.abspath(__file__))

# =============================================== Cr 元素 ===============================================
# %% 
flag = 0
# flag = 1
if flag == 1:
    # 原始谱图.
    Orgin_DATAPATH = os.path.join(ROOTPATH, "Cr", "orgin.mat")
    datas = scio.loadmat(Orgin_DATAPATH)
    samples = datas['contents']
    # values = np.ones((len(samples), ), dtype=np.uint8)
    # samples = np.insert(samples, 0, values=values, axis=1)
    samples = pd.DataFrame(samples, columns=["Component", "Content(%)"])
    samples['Curve type'] = "Original R2=0.905850"
    # sns.lmplot(x="Component", y="Content(%)", data=samples, x_estimator=np.mean)
    # plt.show()

    # %
    # 迭代小波
    IW_DATAPATH = os.path.join(ROOTPATH, "Cr", "wavelet.mat")
    IW_datas = scio.loadmat(IW_DATAPATH)
    IW_samples = IW_datas['contents_wavelet']
    # values = np.ones((len(IW_samples), ), dtype=np.uint8) * 2
    # IW_samples = np.insert(IW_samples, 0, values=values, axis=1)
    IW_samples = pd.DataFrame(IW_samples, columns=["Component", "Content(%)"])
    IW_samples['Curve type'] = "Iterative wavelet R2=0.943579"
    # sns.lmplot(x="Component", y="Content(%)", data=IW_samples, x_estimator=np.mean)

    # %
    # 改进窗迭代自适应小波和高斯卷积
    IIWWG_DATAPATH = os.path.join(ROOTPATH, "Cr", "IIW_wave_gass_not_mom.mat")
    IIWWG_datas = scio.loadmat(IIWWG_DATAPATH)
    IIWWG_samples = IIWWG_datas['contents_IIW_wave_gass']
    # values = np.ones((len(IIWWG_samples), ), dtype=np.uint8) * 3
    # IIWWG_samples = np.insert(IIWWG_samples, 0, values=values, axis=1)
    IIWWG_samples = pd.DataFrame(IIWWG_samples, columns=["Component", "Content(%)"])
    IIWWG_samples['Curve type'] = "AWIAWT-GC R2=0.987406 "
    # sns.lmplot(x="Component", y="Content(%)", data=IIWWG_samples, x_estimator=np.mean)

    # %
    # 改进窗迭代自适应小波和高斯卷积(引入momentum)
    IIWWGm_DATAPATH = os.path.join(ROOTPATH, "Cr", "IIW_wave_gass.mat")
    IIWWGm_datas = scio.loadmat(IIWWGm_DATAPATH)
    IIWWGm_samples = IIWWGm_datas['contents_IIW_wave_gass']
    # values = np.ones((len(IIWWGm_samples), ), dtype=np.uint8) * 3
    # IIWWGm_samples = np.insert(IIWWGm_samples, 0, values=values, axis=1)
    IIWWGm_samples = pd.DataFrame(IIWWGm_samples, columns=["Component", "Content(%)"])
    IIWWGm_samples['Curve type'] = "AWIAWT-GC(momentum) R2=0.987609 "
    # sns.lmplot(x="Component", y="Content(%)", data=IIWWGm_samples, x_estimator=np.mean)

    # %
    # samples = np.vstack((samples, IW_samples, IIWWG_samples))
    # samples = pd.DataFrame(samples, columns=["class", "Component", "Content(%)"])
    df = pd.concat([samples, IW_samples, IIWWG_samples, IIWWGm_samples])

    # %
    g = sns.lmplot(x="Component", y="Content(%)", hue="Curve type", data=df, x_estimator=np.mean, height=7, markers=["o", "s", "x", "v"], legend_out=False)
    ax = g.axes.flat[0]
    ax.set_title(label='Cr', fontsize=16, family='Times New Roman', weight='bold')
    ax.set_xlabel(xlabel="Component", fontsize=16, family='Times New Roman', weight='bold')
    ax.set_ylabel(ylabel="Content(%)", fontsize=16, family='Times New Roman', weight='bold')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig('./Cr.png', dpi=2800, bbox_inches='tight')

# =============================================== Mn 元素 ===============================================
# %% 
flag = 0
# flag = 1
if flag == 1:
    # 原始谱图.
    Orgin_DATAPATH_Mn = os.path.join(ROOTPATH, "Mn", "orgin.mat")
    datas_Mn = scio.loadmat(Orgin_DATAPATH_Mn)
    samples_Mn = datas_Mn['contents']
    # values = np.ones((len(samples), ), dtype=np.uint8)
    # samples = np.insert(samples, 0, values=values, axis=1)
    samples_Mn = pd.DataFrame(samples_Mn, columns=["Component", "Content(%)"])
    samples_Mn['Curve type'] = "Original R2=0.882660"
    # sns.lmplot(x="Component", y="Content(%)", data=samples, x_estimator=np.mean)
    # plt.show()

    # %
    # 迭代小波
    IW_DATAPATH_Mn = os.path.join(ROOTPATH, "Mn", "wavelet.mat")
    IW_datas_Mn = scio.loadmat(IW_DATAPATH_Mn)
    IW_samples_Mn = IW_datas_Mn['contents_wavelet']
    # values = np.ones((len(IW_samples), ), dtype=np.uint8) * 2
    # IW_samples = np.insert(IW_samples, 0, values=values, axis=1)
    IW_samples_Mn = pd.DataFrame(IW_samples_Mn, columns=["Component", "Content(%)"])
    IW_samples_Mn['Curve type'] = "Iterative wavelet R2=0.941452"
    # sns.lmplot(x="Component", y="Content(%)", data=IW_samples, x_estimator=np.mean)

    # %
    # 改进窗迭代自适应小波和高斯卷积
    IIWWG_DATAPATH_Mn = os.path.join(ROOTPATH, "Mn", "IIW_wave_gass_not_mom.mat")
    IIWWG_datas_Mn = scio.loadmat(IIWWG_DATAPATH_Mn)
    IIWWG_samples_Mn = IIWWG_datas_Mn['contents_IIW_wave_gass']
    # values = np.ones((len(IIWWG_samples), ), dtype=np.uint8) * 3
    # IIWWG_samples = np.insert(IIWWG_samples, 0, values=values, axis=1)
    IIWWG_samples_Mn = pd.DataFrame(IIWWG_samples_Mn, columns=["Component", "Content(%)"])
    IIWWG_samples_Mn['Curve type'] = "AWIAWT-GC R2=0.963891"
    # sns.lmplot(x="Component", y="Content(%)", data=IIWWG_samples, x_estimator=np.mean)

    # %
    # 改进窗迭代自适应小波和高斯卷积
    IIWWGm_DATAPATH_Mn = os.path.join(ROOTPATH, "Mn", "IIW_wave_gass.mat")
    IIWWGm_datas_Mn = scio.loadmat(IIWWGm_DATAPATH_Mn)
    IIWWGm_samples_Mn = IIWWGm_datas_Mn['contents_IIW_wave_gass']
    # values = np.ones((len(IIWWG_samples), ), dtype=np.uint8) * 3
    # IIWWG_samples = np.insert(IIWWG_samples, 0, values=values, axis=1)
    IIWWGm_samples_Mn = pd.DataFrame(IIWWGm_samples_Mn, columns=["Component", "Content(%)"])
    IIWWGm_samples_Mn['Curve type'] = "AWIAWT-GC(momentum) R2=0.963999"
    # sns.lmplot(x="Component", y="Content(%)", data=IIWWG_samples, x_estimator=np.mean)

    # %
    # samples = np.vstack((samples, IW_samples, IIWWG_samples))
    # samples = pd.DataFrame(samples, columns=["class", "Component", "Content(%)"])
    df_Mn = pd.concat([samples_Mn, IW_samples_Mn, IIWWG_samples_Mn, IIWWGm_samples_Mn])

    # %
    g = sns.lmplot(x="Component", y="Content(%)", hue="Curve type", data=df_Mn, x_estimator=np.mean, height=7, markers=["o", "s", "x", "v"], legend_out=False)
    ax = g.axes.flat[0]
    ax.set_title(label='Mn', fontsize=16, family='Times New Roman', weight='bold')
    ax.set_xlabel(xlabel="Component", fontsize=16, family='Times New Roman', weight='bold')
    ax.set_ylabel(ylabel="Content(%)", fontsize=16, family='Times New Roman', weight='bold')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig('./Mn.png', dpi=2800, bbox_inches='tight')

# =============================================== Cu 元素 ===============================================
# %% 
flag = 0
# flag = 1
if flag == 1:
    # 原始谱图.
    Orgin_DATAPATH_Cu = os.path.join(ROOTPATH, "Cu", "orgin.mat")
    datas_Cu = scio.loadmat(Orgin_DATAPATH_Cu)
    samples_Cu = datas_Cu['contents']
    # values = np.ones((len(samples), ), dtype=np.uint8)
    # samples = np.insert(samples, 0, values=values, axis=1)
    samples_Cu = pd.DataFrame(samples_Cu, columns=["Component", "Content(%)"])
    samples_Cu['Curve type'] = "Original R2=0.883195"
    # sns.lmplot(x="Component", y="Content(%)", data=samples, x_estimator=np.mean)
    # plt.show()

    # %
    # 迭代小波
    IW_DATAPATH_Cu = os.path.join(ROOTPATH, "Cu", "wavelet.mat")
    IW_datas_Cu = scio.loadmat(IW_DATAPATH_Cu)
    IW_samples_Cu = IW_datas_Cu['contents_wavelet']
    # values = np.ones((len(IW_samples), ), dtype=np.uint8) * 2
    # IW_samples = np.insert(IW_samples, 0, values=values, axis=1)
    IW_samples_Cu = pd.DataFrame(IW_samples_Cu, columns=["Component", "Content(%)"])
    IW_samples_Cu['Curve type'] = "Iterative wavelet R2=0.940194"
    # sns.lmplot(x="Component", y="Content(%)", data=IW_samples, x_estimator=np.mean)

    # %
    # 改进窗迭代自适应小波和高斯卷积
    IIWWG_DATAPATH_Cu = os.path.join(ROOTPATH, "Cu", "IIW_wave_gass_not_mom.mat")
    IIWWG_datas_Cu = scio.loadmat(IIWWG_DATAPATH_Cu)
    IIWWG_samples_Cu = IIWWG_datas_Cu['contents_IIW_wave_gass']
    # values = np.ones((len(IIWWG_samples), ), dtype=np.uint8) * 3
    # IIWWG_samples = np.insert(IIWWG_samples, 0, values=values, axis=1)
    IIWWG_samples_Cu = pd.DataFrame(IIWWG_samples_Cu, columns=["Component", "Content(%)"])
    IIWWG_samples_Cu['Curve type'] = "AWIAWT-GC R2=0.978141"
    # sns.lmplot(x="Component", y="Content(%)", data=IIWWG_samples, x_estimator=np.mean)

    # %
    # 改进窗迭代自适应小波和高斯卷积(含动量)
    IIWWGm_DATAPATH_Cu = os.path.join(ROOTPATH, "Cu", "IIW_wave_gass.mat")
    IIWWGm_datas_Cu = scio.loadmat(IIWWGm_DATAPATH_Cu)
    IIWWGm_samples_Cu = IIWWGm_datas_Cu['contents_IIW_wave_gass']
    # values = np.ones((len(IIWWG_samples), ), dtype=np.uint8) * 3
    # IIWWG_samples = np.insert(IIWWG_samples, 0, values=values, axis=1)
    IIWWGm_samples_Cu = pd.DataFrame(IIWWGm_samples_Cu, columns=["Component", "Content(%)"])
    IIWWGm_samples_Cu['Curve type'] = "AWIAWT-GC(momentum) R2=0.980205"
    # sns.lmplot(x="Component", y="Content(%)", data=IIWWG_samples, x_estimator=np.mean)

    # %
    # samples = np.vstack((samples, IW_samples, IIWWG_samples))
    # samples = pd.DataFrame(samples, columns=["class", "Component", "Content(%)"])
    df_Cu = pd.concat([samples_Cu, IW_samples_Cu, IIWWG_samples_Cu, IIWWGm_samples_Cu])

    # %
    g = sns.lmplot(x="Component", y="Content(%)", hue="Curve type", data=df_Cu, x_estimator=np.mean, height=7, markers=["o", "s", "x", "v"], legend_out=False)
    ax = g.axes.flat[0]
    ax.set_title(label='Cu', fontsize=16, family='Times New Roman', weight='bold')
    ax.set_xlabel(xlabel="Component", fontsize=16, family='Times New Roman', weight='bold')
    ax.set_ylabel(ylabel="Content(%)", fontsize=16, family='Times New Roman', weight='bold')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig('./Cu.png', dpi=2800, bbox_inches='tight')

# %%
# ========================================== 绘制本底扣除效果图 =======================================
sns.set_style("whitegrid")
energy_p = os.path.join(ROOTPATH, "Background", "energy.mat") # 原信号.
background_wavelet_p = os.path.join(ROOTPATH, "Background", "background_wavelet.mat") # 迭代小波本底扣除.
background_IIWWG_p = os.path.join(ROOTPATH, "Background", "background_IIW_wave_Gass_momentum.mat") # 不含动量
background_IIWWGm_p = os.path.join(ROOTPATH, "Background", "background_IIW_wave_Gass.mat") # 含动量.

energy = scio.loadmat(energy_p)
background_wavelet = scio.loadmat(background_wavelet_p)
background_IIWWG = scio.loadmat(background_IIWWG_p)
background_IIWWGm = scio.loadmat(background_IIWWGm_p)

energy = energy["energy"] # (59, 2048)
background_wavelet = background_wavelet["background_wavelet"]
background_IIWWG = background_IIWWG["background_IIW_wave_Gass"]
background_IIWWGm = background_IIWWGm["background_IIW_wave_Gass"]
# %%
flag = 0
# flag = 1
if flag == 1:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    x = [i+1 for i in range(1500)]
    ax.plot(x, energy[0, 0:1500], linewidth=1, label="Original")
    ax.plot(x, background_wavelet[0, 0:1500], linewidth=1, label="Iterative wavelet")
    ax.plot(x, background_IIWWG[0, 0:1500], linewidth=1, label="AWIAWT-GC")
    ax.plot(x, background_IIWWGm[0, 0:1500], linewidth=1, label="AWIAWT-GC(momentum)")
    ax.legend()
    ax.set_xlabel("Aisle", fontsize=16, family='Times New Roman', weight='bold')
    ax.set_ylabel("Count value", fontsize=16, family='Times New Roman', weight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.savefig('./Original spectrum.png', dpi=1200, bbox_inches='tight')

# %%
# flag = 0
flag = 1
if flag == 1:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    x = [i+1 for i in range(2048)]
    ax.plot(x, energy[0, :], linewidth=1, label="Original")
    ax.plot(x, background_wavelet[0, :], linewidth=1, label="Iterative wavelet")
    ax.plot(x, background_IIWWG[0, :], linewidth=1, label="AWIAWT-GC")
    ax.plot(x, background_IIWWGm[0, :], linewidth=1, label="AWIAWT-GC(momentum)")
    ax.legend()
    ax.set_xlabel("Aisle", fontsize=16, family='Times New Roman', weight='bold')
    ax.set_ylabel("Count value", fontsize=16, family='Times New Roman', weight='bold')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()
    # plt.savefig('./local spectrum.png', dpi=2800, bbox_inches='tight')

# %%
