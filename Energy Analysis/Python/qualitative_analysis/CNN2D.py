# 利用二维卷积神经网络，建立能谱分类模型，判别金属土壤大类
# Author:
#       何星华
# History:
#       2020.9.24


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
target = {'Mg_a':1.25, 'Mg_b':1.3, 'Al_a':1.49, 'Al_b':1.55, 'Si_a':1.74, 'Si_b':1.838, 'As_a':10.532, 'As_b':11.729, 
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
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            energy.append(temp)
            m += 3
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
        # count_sum = 0
        for _ in range(2048):
            temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
            # count_sum = count_sum + temp
            energy.append([temp])
            m += 3

    # for i in range(len(energy)):
    #     energy[i][0] = energy[i][0] / count_sum

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

# %% 数据降维处理
x_dataset = np.zeros(shape=(len(alloyDatas)+len(lightAlloyDatas)+len(soilDatas), len(target), 20))
y_dataset = np.zeros(shape=(len(alloyDatas)+len(lightAlloyDatas)+len(soilDatas),))

# %%
# 处理普通金属
for i in range(len(alloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        if aisle < x_dataset.shape[-1]:
            x_dataset[i][j][0:aisle] = alloyDatas[i][0:aisle]
            x_dataset[i][j][aisle:x_dataset.shape[-1]] = alloyDatas[i][aisle:x_dataset.shape[-1]]
        else:
            x_dataset[i][j][0:x_dataset.shape[-1]] = alloyDatas[i][aisle-10:aisle+10]
        j += 1
    y_dataset[i] = 0
# 处理轻金属
for i in range(len(lightAlloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        if aisle < x_dataset.shape[-1]:
            x_dataset[i+len(alloyDatas)][j][0:aisle] = alloyDatas[i][0:aisle]
            x_dataset[i+len(alloyDatas)][j][aisle:x_dataset.shape[-1]] = alloyDatas[i][aisle:x_dataset.shape[-1]]
        else:
            x_dataset[i+len(alloyDatas)][j][0:x_dataset.shape[-1]] = alloyDatas[i][aisle-10:aisle+10]
        j += 1
    y_dataset[i+len(alloyDatas)] = 1
# 处理土壤
for i in range(len(soilDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        if aisle < x_dataset.shape[-1]:
            x_dataset[i+len(alloyDatas)+len(lightAlloyDatas)][j][0:aisle] = soilDatas[i][0:aisle]
            x_dataset[i+len(alloyDatas)+len(lightAlloyDatas)][j][aisle:x_dataset.shape[-1]] = soilDatas[i][aisle:x_dataset.shape[-1]]
        else:
            x_dataset[i+len(alloyDatas)+len(lightAlloyDatas)][j][0:x_dataset.shape[-1]] = soilDatas[i][aisle-10:aisle+10]
        j += 1
    y_dataset[i+len(alloyDatas)+len(lightAlloyDatas)] = 2

# =============================================================================
# ======================== 将数据转化为图片 ====================================
# =============================================================================
# %%
# 将特征转化为图像分别保存.
for i in range(len(x_dataset)):
    if i < len(alloyDatas):
        plt.imshow(x_dataset[i])
        plt.savefig('./images/alloy/{0}_label_{1}.png'.format(i, int(y_dataset[i])))
    elif len(alloyDatas) <= i < len(lightAlloyDatas) + len(alloyDatas):
        plt.imshow(x_dataset[i])
        plt.savefig('./images/lightalloy/{0}_label_{1}.png'.format(i, int(y_dataset[i])))
    elif len(alloyDatas) + len(lightAlloyDatas) <= i:
        plt.imshow(x_dataset[i])
        plt.savefig('./images/soil/{0}_label_{1}.png'.format(i, int(y_dataset[i])))

# =====================================================================
# ============================  二维卷积神经网络 ======================
# ===================================================================
# %%
# 读取图片数据
data_dir = './images'
train_tfrecord_file = data_dir + '/tfrecord/train.tfrecords'
train_alloy_dir = data_dir + '/alloy/'
train_lightalloy_dir = data_dir + '/lightalloy/'
train_soil_dir = data_dir + '/soil/'

# %%
train_alloy_filenames = [train_alloy_dir + filename for filename in os.listdir(train_alloy_dir)]
train_lightalloy_filenames = [train_lightalloy_dir + filename for filename in os.listdir(train_lightalloy_dir)]
train_soil_filenames = [train_soil_dir + filename for filename in os.listdir(train_soil_dir)]

train_filenames = train_alloy_filenames + train_lightalloy_filenames + train_soil_filenames
train_labels = [0] * len(train_alloy_filenames) + [1] * len(train_lightalloy_filenames) +[2] * len(train_soil_filenames)

# %%
with tf.io.TFRecordWriter(train_tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()

        feature = {
            'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

# %%
# 读取数据
train_dataset = tf.data.TFRecordDataset(train_tfrecord_file)
feature_description = {
    'image' : tf.io.FixedLenFeature([], tf.string),
    'label' : tf.io.FixedLenFeature([], tf.int64)
}

def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']


train_dataset = train_dataset.map(_parse_example)

batch_size = 32

train_dataset = train_dataset.shuffle(buffer_size=1024)    
train_dataset = train_dataset.batch(batch_size)

class CNNModel(tf.keras.models.Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)       
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

# %%
learning_rate = 0.001
model = CNNModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#batch
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss) #update
    train_accuracy(labels, predictions)#update

# %%
EPOCHS=100
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(images, labels) #mini-batch 更新

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                         ))


# ===================================================================================
# ================================== 旧 ============================================
# ===================================================================================
# %%
# 标准化
# x_dataset = scale(x_dataset)
# i = 0
# for data in x_dataset:
#     x_dataset[i] = scale(x_dataset[i])
#     i += 1
x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(x_dataset, y_dataset, test_size=0.4, random_state=8)

# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(len(target), 20, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation=None))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=None))
model.add(layers.BatchNormalization())
model.add(layers.Dense(3, activation='softmax'))

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# %%
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dataset, y_train_dataset)).shuffle(1024).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_dataset, y_test_dataset)).batch(4)

# %%
# log_dir="logs/cnn2d_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# model.fit(train_dataset, epochs=10000, validation_data=test_dataset, shuffle=True, callbacks=[tensorboard_callback])
model.fit(train_dataset, epochs=100, validation_data=test_dataset, shuffle=True)

# %%
model.summary()
# %%
print(y_test_dataset)
# %%
