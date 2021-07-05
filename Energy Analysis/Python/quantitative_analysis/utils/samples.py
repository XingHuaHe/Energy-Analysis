from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import scipy.io as scio

def get_samples(path):
    datas = scio.loadmat(path)
    # x.
    samples = datas['date']
    feature1 = samples[:, 0]
    feature2 = samples[:, 1]
    feature3 = samples[:, 8]
    features = np.vstack((feature1, feature2, feature3)).transpose()
    xScale = preprocessing.StandardScaler()
    yScale = preprocessing.StandardScaler()
    samples = xScale.fit_transform(features)
    # y.
    standard_content = datas['result']
    temp = np.zeros((standard_content.shape[0],))
    for i in range(temp.shape[0]):
        temp[i] = standard_content[i][23]
    contents =  temp # yScale.fit_transform(np.reshape(temp, (-1, 1)))

    x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = train_test_split(samples, contents, test_size=0.2, random_state=3)
    y_train_dataset = yScale.fit_transform(np.reshape(y_train_dataset, (-1, 1)))
    y_test_dataset1 = yScale.fit_transform(np.reshape(y_test_dataset, (-1, 1)))

    train_dataset = list()
    test_dataset = list()
    for i in range(len(x_train_dataset)):
        train_dataset.append((np.array(x_train_dataset[i]), np.array(y_train_dataset[i])))
    for i in range(len(x_test_dataset)):
        test_dataset.append((np.array(x_test_dataset[i]), np.array(y_test_dataset1[i])))

    return (train_dataset, test_dataset, y_test_dataset, yScale)