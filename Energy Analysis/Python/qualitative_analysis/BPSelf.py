# %% 导入库
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import tensorflow as tf

# %% 初始化参数 y = ax + b （普通合金）
a = 0.0347004803616841
b = -0.0398474145238765

target = {'Ti':4.51, 'V':4.95, 'Cr':5.41, 'Mn':5.895, 'Fe':6.4,
          'Co':6.925, 'Ni':7.47, 'Cu':8.04, 'Zu':8.63,
          'Zr':15.746, 'Nb':16.6584, 'Mo':17.443, 'Ag':22.1,
          'Sn':25.193}

# %% 读取数据
# 合金数据
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
            temp = data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]
            energy.append(temp)
            m += 3
    alloyDatas.append(energy)
alloyDatas = np.array(alloyDatas)

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
            temp = data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]
            energy.append(temp)
            m += 3
    soilDatas.append(energy)
soilDatas = np.array(soilDatas)

# %% 数据降维处理
x_dataset = np.zeros(shape=(len(alloyDatas)+len(soilDatas), len(target)))
y_dataset = np.zeros(shape=(len(alloyDatas)+len(soilDatas),))
# 处理金属
for i in range(len(alloyDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i][j] = alloyDatas[i][aisle]
        j += 1
    y_dataset[i] = 0
# 处理土壤
for i in range(len(soilDatas)):
    j = 0
    for value in target.values():
        aisle = int((value - b) / a)
        x_dataset[i+len(alloyDatas)][j] = soilDatas[i][aisle]
        j += 1
    y_dataset[i+len(alloyDatas)] = 1

x_dataset = scale(x_dataset)
x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=8)

# %% 自定义层
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        self.units = units
        super(MyDense, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='w')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='b')
        super(MyDense, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(MyDense, self).get_config()
        config.update({'units':self.units})
        return config

# %% 自定义模型
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=1):
        super(MyModel, self).__init__(name='my_model')

        self.num_classes = num_classes
        self.MyDense1 = MyDense(units=128)
        self.MyDense2 = MyDense(units=128)
        self.MyDense3 = MyDense(units=64)
        self.MyDense4 = MyDense(units=self.num_classes)
    
    def call(self, inputs):
        x = self.MyDense1(inputs)
        x = tf.nn.relu(x)
        x = self.MyDense2(x)
        # x = tf.keras.layers.BatchNormalization()
        x = self.MyDense3(x)
        tf.keras.layers.Dropout(rate=0.2).apply(x)
        x = tf.nn.relu(x)
        model = self.MyDense4(x)
        return model


# %% 自定义损失函数
class MySparcesCategoricalCrossentropyLoss(tf.keras.losses.Loss):
    def __init__(self, name=None):
        super(MySparcesCategoricalCrossentropyLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        # y_true is sparce code.
        y_pred = tf.nn.softmax(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true = tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1])
        y_true = tf.cast(y_true, tf.float32)

        loss = - y_true * tf.math.log(y_pred)
        loss = tf.reduce_sum(loss, axis=1)

        return loss

# %% 自定义指标函数
class MySparcesCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='MySparcesCategoricalAccuracy', **kwargs):
        super(MySparcesCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.float32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        value = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.count.assign_add(tf.cast(tf.reduce_sum(value), tf.float32))

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)

# %% 参数初始化
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dataset, y_train_dataset)).shuffle(1024).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_dataset, y_test_dataset)).batch(1)

model = MyModel(num_classes=2)
# 损失函数
losser = MySparcesCategoricalCrossentropyLoss()
# 优化器
optimizer = tf.keras.optimizers.Adam()
# 评价指标
train_acc_metric = MySparcesCategoricalAccuracy()
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
test_acc_metric = MySparcesCategoricalAccuracy()
# test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

# %%
def train(EPOCHS):
    for epoch in range(EPOCHS):
        template = 'Epoch {}, Training Loss: {}, Training Accuracy: {}%, Testing Accuracy: {}%'

        # Training.
        for step, (x_train, y_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y = model(x_train)
                losses = losser(y_train, y)
            grad = tape.gradient(losses, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            train_acc_metric(y_train, y)
            train_loss_metric(losses)

        # Testing.
        for x_test, y_test in test_dataset:
            val_logits = model(x_test)
            test_acc_metric(y_test, val_logits)
            # test_loss_metric()

        print(template.format(epoch+1, train_loss_metric.result(), train_acc_metric.result()*100, test_acc_metric.result()*100))
        train_acc_metric.reset_states()
        train_loss_metric.reset_states()
        test_acc_metric.reset_states()

# %%
EPOCHS = 100
train(EPOCHS)

# %%
model.summary()
# %%
print(y_test_dataset)