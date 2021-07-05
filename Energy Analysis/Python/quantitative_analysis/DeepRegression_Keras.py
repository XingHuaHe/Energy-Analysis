
'''
# tensorflow keras 版 基于BP神经网络进行定量分析.
'''

# System packages.
import os
import datetime
import argparse

# Extend packages.
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=550, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--data_path", type=str, default=None, help="data path of dataset")
    parser.add_argument("--pretraining", type=bool, default=False, help="pretraining model path")
    parser.add_argument("--pretraining_path", type=str, default="./config/resnet50-19c8e357.pth", help="pretraining model path")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    opt = parser.parse_args()
    print(opt)

    os.makedirs('checkpoints_tf', exist_ok=True)
    os.makedirs('./output/Keras', exist_ok=True) # save output images.
    os.makedirs('./result/Keras', exist_ok=True) # save result data.

    if opt.data_path is not None:
        # Setting data path according args.
        DATAPATH = opt.data_path
    else:
        # Current directory path.
        ROOTPATH = os.path.dirname(os.path.abspath(__file__))
        # Data path.
        DATAPATH = os.path.join(ROOTPATH, "..", "DatDatas", "features.mat")

    # obtain features.
    datas = scio.loadmat(DATAPATH)
    samples = datas['features']
    features = samples[:, 0:10]
    # obtain content.
    standard_content = samples[:, 10:13]

    # Standar features.
    xScale = preprocessing.StandardScaler()
    features = xScale.fit_transform(features, standard_content)
    # Standar contents.
    yScale = preprocessing.StandardScaler()
    standard_content = yScale.fit_transform(standard_content)

    # split dataset.
    x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(features, standard_content, test_size=0.2, random_state=4, shuffle=True)

    # Defined model.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(10,)),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(3, activation='tanh')
    ])

    # Defined optimizer.
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=opt.lr)

    # Setting model.
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])

    # Construct Dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dataset, y_train_dataset)).shuffle(1024).batch(4)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_dataset, y_test_dataset)).batch(1)

    # Training.
    model.fit(train_dataset, epochs=opt.epochs)

    # predicted.
    target = np.zeros_like(y_test_dataset)
    Cr = list()
    Mn = list()
    Cu = list()
    for i, (x_test, y_test) in enumerate(test_dataset):
        predict = model(x_test)
        predict_sacle = yScale.inverse_transform(predict.numpy()[0]) # <class 'numpy.ndarray'>
        Cr.append(predict_sacle[0]) # <class 'list'>
        Mn.append(predict_sacle[1]) # <class 'list'>
        Cu.append(predict_sacle[2]) # <class 'list'>
        target[i] = yScale.inverse_transform(y_test.numpy()) # <class 'numpy.ndarray'>

    # Plot figures.
    _, ax = plt.subplots()
    ax.plot([i for i in range(len(y_test_dataset))], Cr)
    ax.plot([i for i in range(len(y_test_dataset))], Mn)
    ax.plot([i for i in range(len(y_test_dataset))], Cu)
    ax.plot([i for i in range(len(y_test_dataset))], target[:,0], "--")
    ax.plot([i for i in range(len(y_test_dataset))], target[:,1], "--")
    ax.plot([i for i in range(len(y_test_dataset))], target[:,2], "--")
    plt.savefig(f"./output/keras/{opt.epochs}.jpg")

    # Save data.
    elements = pd.DataFrame({'Cr':Cr, 'Mn':Mn, 'Cu':Cu})
    np.savetxt(f'./result/keras/{opt.epochs}_target.csv', target, fmt='%f', delimiter=',')
    np.savetxt(f'./result/keras/{opt.epochs}_predict_sacle.csv', predict_sacle, fmt='%f', delimiter=',')
    elements.to_csv(f'./result/keras/{opt.epochs}_elements.csv', index=False, sep=',')
