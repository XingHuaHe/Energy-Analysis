# %% 导入库文件
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale

# %% 读取土壤数据
rootpath = os.getcwd()
filepath = os.path.join(rootpath,'DatDatas', '数据-土壤v1', '180207152653170814120321土壤.csv')
GSS = pd.read_csv(filepath)

labels_name = ['F_9', 'Na_11', 'Mg_12', 'Al_13', 'P_15', 'S_16', 'Cr_24',
          'Mn_25', 'Fe_26', 'Co_27', 'Ni_28', 'Cu_29', 'Zn_30']
labels_code = [i for i in range(len(labels_name))]
labels_dict = dict(zip(labels_code, labels_name))
dataset = scale(GSS[labels_name].to_numpy())
# dataset = GSS[labels_name].to_numpy()
labels = np.ones(shape=(len(dataset),)) * 3

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
        self.MyDense1 = MyDense(units=64)
        self.MyDense2 = MyDense(units=32)
        self.MyDense3 = MyDense(units=self.num_classes)

    def call(self, inputs):
        x = self.MyDense1(inputs)
        x = self.MyDense2(x)
        model = self.MyDense3(x)
        return model

# %% 自定义损失函数
class MySparcesCategoricalCrossentropyLoss(tf.keras.losses.Loss):
    def __init__(self, name=None):
        super(MySparcesCategoricalCrossentropyLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true = tf.one_hot(tf.cast(y_true, tf.int32), 13)
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
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.count.assign_add(tf.cast(tf.reduce_sum(values), tf.float32))

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)

# %% 数据预处理
train_dataset = tf.data.Dataset.from_tensor_slices((dataset, labels)).shuffle(1024).batch(8)

model = MyModel(num_classes=len(labels_code))
losser = MySparcesCategoricalCrossentropyLoss()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = MySparcesCategoricalAccuracy()
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')

# %%
def train(EPOCHS):
    for epoch in range(EPOCHS):
        template = 'Epoch {}, Loss: {}, Accuracy: {}%'

        for step, (x_train, y_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y = model(x_train)
                losses = losser(y_train, y)
            grad = tape.gradient(losses, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            train_acc_metric(y_train, y)
            train_loss_metric(losses)

        print(template.format(epoch+1, train_loss_metric.result(), train_acc_metric.result()*100))
        train_acc_metric.reset_states()
        train_loss_metric.reset_states()

# %%
EPOCHS = 100
train(EPOCHS)

# %%
print(len(labels_name))
# %%
