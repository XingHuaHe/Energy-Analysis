# 利用BP神经网络，建立能谱分类模型，判别金属土壤大类
# Author:
#       Xinghua.He
# History:
#       2020.9.22


# %% 导入库
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import tensorflow as tf

# %%
a = 0.0347004803616841
b = -0.0398474145238765

# target = {'Al':1.49, 'Si':1.74, 'As':10.532, 'K':3.31, 'Ca':3.69, 'Ti':4.51,
#           'V':4.95, 'Cr':5.41, 'Mn':5.895, 'Fe':6.4, 'Co':6.925, 'Ni':7.47, 
#           'Cu':8.04, 'Zu':8.63, 'Zr':15.746, 'Nb':16.6584, 'Mo':17.443, 
#           'Ag':22.1, 'Sn':25.193, 'Ta':8.145, 'W':8.396}
target = {'Al_a':1.49, 'Al_b':1.55, 'Si_a':1.74, 'Si_b':1.838, 'As_a':10.532, 'As_b':11.729, 
          'K_a':3.31, 'K_b':3.59, 'Ca_a':3.69, 'Ca_b':4.01, 'Ti_a':4.51, 'Ti_b':4.93,
          'V_a':4.95, 'V_b':5.43, 'Cr_a':5.41, 'Cr':5.95, 'Mn_a':5.895, 'Mn_b':6.49, 
          'Fe_a':6.4, 'Fe_b':7.06, 'Co_a':6.925, 'Co_b':7.65, 'Ni_a':7.47, 'Ni_b':8.265, 
          'Cu_a':8.04, 'Cu_b':8.907, 'Zn_a':8.63, 'Zn_b':9.572, 'Zr_a':15.746, 'Zr_b':17.687, 
          'Nb_a':16.6584, 'Nb_b':18.647, 'Mo_a':17.443, 'Mo_b':19.633, 
          'Ag_a':22.1, 'Ag_b':24.987, 'Sn_a':25.193, 'Sn_b':28.601, 'Ta_a':8.145, 'Ta_b':9.341, 
          'W_a':8.396, 'W_b':9.67}

# %% 读取数据
# 普通合金数据
# rootpath = os.getcwd()
rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
filespath = os.path.join(rootpath, 'DatDatas', '数据-普通合金')
filenames = os.listdir(filespath)

alloyDatas = []
for filename in filenames:
    if os.path.splitext(filename)[-1].lower() != '.dat':
        continue
    filepath = os.path.join(filespath, filename)
    with open(filepath, 'rb') as f:
        data = f.read()
        energy = []
        m = 5
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            energy.append(temp)
            m += 3
    alloyDatas.append(energy)
alloyDatas = np.array(alloyDatas)
# %%
# # 归一化
# for i in range(len(alloyDatas)):
#     alloyDatas[i] = alloyDatas[i] / np.sum(alloyDatas[i])

# 轻合金
filespath = os.path.join(rootpath, 'DatDatas', '数据-轻合金')
filenames = os.listdir(filespath)

lightAlloyDatas = []
for filename in filenames:
    if os.path.splitext(filename)[-1].lower() != '.dat':
        continue
    filepath = os.path.join(filespath, filename)
    with open(filepath, 'rb') as f:
        data = f.read()
        energy = []
        m = 5
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            energy.append(temp)
            m += 3
    lightAlloyDatas.append(energy)
lightAlloyDatas = np.array(lightAlloyDatas)

# 土壤数据
filespath = os.path.join(rootpath, 'DatDatas', '数据-土壤')
filenames = os.listdir(filespath)

soilDatas = []
for filename in filenames:
    if os.path.splitext(filename)[-1].lower() != '.dat':
        continue
    filepath = os.path.join(filespath, filename)
    with open(filepath, 'rb') as f:
        data = f.read()
        energy = []
        m = 5
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 90
            energy.append(temp)
            m += 3
    soilDatas.append(energy)
soilDatas = np.array(soilDatas)
# # 归一化
# for i in range(len(soilDatas)):
#     soilDatas[i] = soilDatas[i] / np.sum(soilDatas[i])

# %% 数据降维处理
x_dataset = np.zeros(shape=(len(alloyDatas)+len(lightAlloyDatas)+len(soilDatas), len(target)))
y_dataset = np.zeros(shape=(len(alloyDatas)+len(lightAlloyDatas)+len(soilDatas),))
# 处理普通合金
for i in range(len(alloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i][j] = alloyDatas[i][aisle]
        j += 1
    y_dataset[i] = 0
# 处理轻合金
for i in range(len(lightAlloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i+len(alloyDatas)][j] = alloyDatas[i][aisle]
        j += 1
    y_dataset[i+len(alloyDatas)] = 1
# 处理土壤
for i in range(len(soilDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i+len(alloyDatas)+len(lightAlloyDatas)][j] = soilDatas[i][aisle]
        j += 1
    y_dataset[i+len(alloyDatas)+len(lightAlloyDatas)] = 2

# %%
# 标准化
for i in range(len(x_dataset)):
    x_dataset[i] = x_dataset[i] / np.sum(x_dataset[i])
x_dataset = scale(x_dataset)
x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=8)

# %%
input1 = tf.keras.Input(shape=(len(target),))
x = tf.keras.layers.Dense(128, activation='relu')(input1)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
predictions = tf.keras.layers.Dense(3)(x)

model = tf.keras.Model(inputs=input1, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# %%
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dataset, y_train_dataset)).shuffle(1024).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_dataset, y_test_dataset)).batch(4)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# %%
model.fit(train_dataset, epochs=1000, validation_data=test_dataset, shuffle=True, callbacks=[tensorboard_callback])
# %%
model.summary()
# %%
