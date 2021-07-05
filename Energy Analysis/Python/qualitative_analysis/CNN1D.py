# 利用一维卷积神经网络，建立能谱分类模型，判别金属土壤大类
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
from tensorflow.keras import layers, models, datasets

# %%
a = 0.0347004803616841
b = -0.0398474145238765

# target = {'Ti':4.51, 'V':4.95, 'Cr':5.41, 'Mn':5.895, 'Fe':6.4,
#           'Co':6.925, 'Ni':7.47, 'Cu':8.04, 'Zu':8.63,
#           'Zr':15.746, 'Nb':16.6584, 'Mo':17.443, 'Ag':22.1,
#           'Sn':25.193}
# 
# 考虑Ka和Kb峰，没有Ka和Kb的，考虑La和Lb峰
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
rootpath = os.getcwd()
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
        count_sum = 0
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            count_sum = count_sum + temp
            energy.append([temp])
            m += 3

    for i in range(len(energy)):
        energy[i][0] = energy[i][0] / count_sum

    alloyDatas.append(energy)
alloyDatas = np.array(alloyDatas)

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
        count_sum = 0
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            count_sum = count_sum + temp
            energy.append([temp])
            m += 3

    for i in range(len(energy)):
        energy[i][0] = energy[i][0] / count_sum

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
        count_sum = 0
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 90
            count_sum = count_sum + temp
            energy.append([temp])
            m += 3
    
    for i in range(len(energy)):
        energy[i][0] = energy[i][0] / count_sum
    
    soilDatas.append(energy)
soilDatas = np.array(soilDatas)

# %% 数据降维处理
x_dataset = np.zeros(shape=(len(alloyDatas)+len(lightAlloyDatas)+len(soilDatas), len(target), 1))
y_dataset = np.zeros(shape=(len(alloyDatas)+len(lightAlloyDatas)+len(soilDatas),))

# %%
# 处理普通合金数据
for i in range(len(alloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i][j][0] = alloyDatas[i][aisle][0]
        j += 1
    y_dataset[i] = 0

# 处理轻合金数据
for i in range(len(lightAlloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i+len(alloyDatas)][j][0] = lightAlloyDatas[i][aisle][0]
        j += 1
    y_dataset[i+len(alloyDatas)] = 1

# 处理土壤
for i in range(len(soilDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i+len(alloyDatas)+len(lightAlloyDatas)][j][0] = soilDatas[i][aisle][0]
        j += 1
    y_dataset[i+len(alloyDatas)+len(lightAlloyDatas)] = 3

# %%
print(x_dataset.shape)

# %%
# 标准化
# x_dataset = scale(x_dataset)

i = 0
for data in x_dataset:
    x_dataset[i] = scale(x_dataset[i])
    i += 1
x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=8)

# %%
# input1 = tf.keras.Input(shape=(None, 14))
# x = layers.Conv1D(32, 3, activation='relu')(input1)
# x = layers.MaxPool1D(2)(x)
# x = layers.Conv1D(64, 3, activation='relu')(x)
# x = layers.MaxPool1D(2)(x)
# x = layers.Flatten()(x)
# x = layers.Dense(64, activation='relu')(x)
# predictions = layers.Dense(2)(x)

model = models.Sequential()
model.add(layers.Conv1D(32, 2, activation='tanh', input_shape=(len(target), 1)))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(64, 2, activation='tanh'))
model.add(layers.Conv1D(128, 2, activation=None))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=None))
model.add(layers.BatchNormalization())
model.add(layers.Dense(3))

# %%
# model = tf.keras.Model(inputs=input1, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

# %%
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dataset, tf.one_hot(tf.cast(y_train_dataset, tf.int32), 3))).shuffle(1024).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_dataset, tf.one_hot(tf.cast(y_test_dataset, tf.int32), 3))).batch(4)

# %%
# callbacks = 
log_dir="logs/cnn_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


model.fit(train_dataset, epochs=100, validation_data=test_dataset, shuffle=True, callbacks=[tensorboard_callback])
# %%
model.summary()
# %%
a = [[1],[2],[3]]
print(a[0][0])
# %%
