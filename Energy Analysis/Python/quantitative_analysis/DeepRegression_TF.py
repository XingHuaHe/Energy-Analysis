'''
# 
# Description:
#       Using BPNN for background deduction.  tensorflow. 基于自定义模型的BP神经网络定量分析.
# Author:
#       Xinghua.He
# History:
#       2021.1.12
# 
# 
'''
# 
from utils.models.Dense import *
from utils.models.LDNN import *
from utils.models.Loss import *

# System packages.
import os
import datetime
import argparse

# Extend packages.
import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--data_path", type=str, default=None, help="data path of dataset")
    parser.add_argument("--pretraining", type=bool, default=False, help="pretraining model path")
    parser.add_argument("--pretraining_path", type=str, default="./config/resnet50-19c8e357.pth", help="pretraining model path")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    opt = parser.parse_args()
    print(opt)

    # Make directory.
    os.makedirs('checkpoints', exist_ok=True) # save checkpoints.
    os.makedirs('./output/tf', exist_ok=True) # save output images.
    os.makedirs('./result/tf', exist_ok=True) # save result data.

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
    # features = xScale.fit_transform(features)
    features = xScale.fit_transform(features, standard_content)
    # Standar contents.
    yScale = preprocessing.StandardScaler()
    standard_content = yScale.fit_transform(standard_content)

    # split dataset.
    x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(features, standard_content, test_size=0.2, random_state=4, shuffle=True)

    # Defined Dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dataset, y_train_dataset)).shuffle(1024).batch(4)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_dataset, y_test_dataset)).batch(1)

    # Defined model.
    model = MyModel(num_classes=3)

    # Defined loss.
    losser = MyLoss()

    # Defined optimizer.
    optimizer = tf.keras.optimizers.RMSprop(opt.lr)

    # Defined mestris.
    # train_acc_metric = MySparcesCategoricalAccuracy()

    # Training.
    for epoch in tqdm.tqdm(range(opt.epochs)):
        # template = 'Epoch {}, Training Loss: {}, Training Accuracy: {}%, Testing Accuracy: {}%'
        # Training.
        for _, (x_train, y_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # y = model(x_train)
                output1, output2, output3 = model(x_train)
                loss1 = losser(y_train, output1)
                loss2 = losser(y_train, output2)
                loss3 = losser(y_train, output3)
                losses = loss1 + loss2 + loss3 
            grad = tape.gradient(losses, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        # Evaluate.
        if (epoch + 1) % 50  == 0 or epoch == 0:
            # predicted.
            target = np.zeros_like(y_test_dataset)
            prediction1 = list()
            prediction2 = list()
            prediction3 = list()
            Cr = list()
            Mn = list()
            Cu = list()
            for i, (x_test, y_test) in enumerate(test_dataset):
                # Forward.
                predict1, predict2, predict3 = model(x_test)

                prediction1.append(yScale.inverse_transform(predict1.numpy()[0])) # <class 'list'>
                prediction2.append(yScale.inverse_transform(predict2.numpy()[0])) # <class 'list'>
                prediction3.append(yScale.inverse_transform(predict3.numpy()[0])) # <class 'list'>
                target[i] = yScale.inverse_transform(y_test.numpy()) # <class 'numpy.ndarray'>
            # Eventral output.
            for i in range(len(y_test_dataset)):
                Cr.append(prediction1[i][0] * 0.3 + prediction2[i][0] * 0.3 + prediction3[i][0] * 0.4)
            for i in range(len(y_test_dataset)):
                Mn.append(prediction1[i][1] * 0.3 + prediction2[i][1] * 0.3 + prediction3[i][1] * 0.4)
            for i in range(len(y_test_dataset)):
                Cu.append(prediction1[i][2] * 0.3 + prediction2[i][2] * 0.3 + prediction3[i][2] * 0.4)
            
            # Plot figures.
            _, ax = plt.subplots()
            ax.plot([i for i in range(len(y_test_dataset))], Cr)
            ax.plot([i for i in range(len(y_test_dataset))], Mn)
            ax.plot([i for i in range(len(y_test_dataset))], Cu)
            ax.plot([i for i in range(len(y_test_dataset))], target[:,0], "--")
            ax.plot([i for i in range(len(y_test_dataset))], target[:,1], "--")
            ax.plot([i for i in range(len(y_test_dataset))], target[:,2], "--")
            plt.savefig(f"./output/tf/{epoch+1}.jpg")

            # Save data.
            predictions = pd.DataFrame({'prediction1':prediction1, 'prediction2':prediction2, 'prediction3':prediction3})
            elements = pd.DataFrame({'Cr':Cr, 'Mn':Mn, 'Cu':Cu})
            np.savetxt(f'./result/tf/{epoch+1}_target.csv', target, fmt='%f', delimiter=',')
            predictions.to_csv(f'./result/tf/{epoch+1}_predictions.csv', index=False, sep=',')
            elements.to_csv(f'./result/tf/{epoch+1}_elements.csv', index=False, sep=',')

            # Release storage.
            del prediction1
            del prediction2
            del prediction3
            del target
            del Cr
            del Cu
            del Mn
            del predictions
            del elements

    # model summary.
    model.summary()
