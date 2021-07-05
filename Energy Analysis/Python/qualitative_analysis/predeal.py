# Convert energy spectrum data from .dat form to .csv  form.
# Author:
#        Xinghua.He
# History:
#       2020.9.18


# %%
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# %%
target = ['Ti_22', 'V_23', 'Cr_24', 'Mn_25', 'Fe_26', 'Co_27', 'Ni_28',
          'Cu_29', 'Zn_30', 'Zr_40', 'Nb_41', 'Mo_42', 'Ag_47', 'Sn_50',
          'Ta_73', 'W_74', 'Pb_82']
rootpath = os.getcwd()
contentTabel = os.path.join(rootpath, 'DatDatas', '数据-普通合金', '普通合金含量表.xls')
df = pd.read_excel(contentTabel)
labels = df[['sample_id']].to_numpy()
labels = np.reshape(labels, (len(labels),))
df = df[target].to_numpy()

# %%
print(type(df[0][0]))



# %%
def dataFormConvert(path):
    '''Convert data format from .dat to .csv.'''
    rootpath = os.getcwd()
    filespath = os.path.join(rootpath, path)
    filespath = [os.path.join(filespath, filepath) for filepath in os.listdir(filespath)]

    for filepath in filespath:
        energyDatas = []
        labels = []
        for filename in os.listdir(filepath):
            if os.path.splitext(filename)[-1].lower() != '.dat':
                continue
            filenamepath = os.path.join(filepath, filename)
            with open(filenamepath, 'rb') as f:
                data = f.read()
                # sampleId = data[3] * 256 + data[4]
                energy = []
                m = 5
                for _ in range(2048):
                    temp = data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]
                    energy.append(temp)
                    m += 3
        

def plotSpectrum():
    rootpath = os.getcwd()
    df = pd.read_csv(os.path.join(rootpath, 'DatDatas', '数据-普通合金.csv'))

if __name__ == "__main__":
    dataFormConvert('DatDatas')


# %%
