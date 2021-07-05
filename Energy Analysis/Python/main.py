# %%
import os
import datetime

import tqdm
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wavelet_background.windows_wbd as wbd

# %
def extract_datas(filepath):
    filenames = os.listdir(filepath)
    alloyDatas = []
    alloyId = []
    for filename in filenames:
        if filename.split('.')[-1].lower() == 'xls':
            labels = pd.read_excel(os.path.join(filepath, filename))
            continue
        elif filename.split('.')[-1].lower() != 'dat':
            continue
        else:
            # extract .dat.
            with open(os.path.join(filepath, filename), 'rb') as f:
                data = f.read()
                energy = []
                m = 5
                for _ in range(2048):
                    temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
                    energy.append(temp)
                    m += 3
                Id = data[3] * 256 + data[4]
            if Id < -1:
                continue
            alloyId.append(Id)
            alloyDatas.append(energy)
    alloyDatas = np.array(alloyDatas)
    alloyId = np.array(alloyId)

    # extract Cu content.
    Cu_s = []
    for i in range(len(alloyId)):
        alloy = labels.loc[labels['sample_id'] == alloyId[i]]
        Cu = alloy['Cr_24'].values[0]
        Cu_s.append(Cu)
    Cu_s = np.array(Cu_s)

    return (alloyDatas, alloyId, Cu_s)

# %
if __name__ == "__main__":
    a = 0.0347004803616841
    b = -0.0398474145238765
    
    # Path.
    # rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    rootpath = os.path.abspath(os.path.dirname("__file__"))
    filepath = os.path.join(rootpath, 'DatDatas', '数据-土壤')

    # Getting energy data.
    alloyDatas, alloyId, Cu_s = extract_datas(filepath)

    # %
    # background deduction.
    for i in range(len(alloyDatas)):
        temp = wbd.background_deduction(alloyDatas[i])
        # en = np.zeros_like(temp)
        # for j in range(len(temp)):
        #     en[j] = temp[j] * a + b
        # alloyDatas[i] = en
        alloyDatas[i] = temp

    # Computed component.
    axis = int((5.41 - b) / a) - 1
    # ag_axis = int((22.10 - b) / a) - 1
    components = np.zeros_like(Cu_s)
    for j in range(len(alloyDatas)):
        area = np.sum(alloyDatas[j][axis-1:axis+1])
        # ag_area = np.sum(alloyDatas[j][ag_axis-10:ag_axis+10])
        # components[j] = area / ag_area
        components[j] = area
    
    # %
    _, ax = plt.subplots()
    ax.plot([i for i in range(2048)], alloyDatas[0], linewidth=0.3, label='o')
    ax.plot([axis, axis], [0, alloyDatas[0][axis]], linewidth=0.3, label='n')
    # ax.plot([ag_axis, ag_axis], [0, alloyDatas[0][ag_axis]], linewidth=0.3, label='n')
    # ax.plot([i for i in range(2048)], alloyDatas[2], linewidth=0.3, label='n1')
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title('simple plot')
    ax.legend()
    plt.show()

    # %
    from scipy import stats
    # fit function: y = ax + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(components, Cu_s)
    print("斜率：", slope,"截距：", intercept, "相关系数：", r_value, "R2：", r_value**2)

    plt.plot(components, Cu_s, 'ro', [0, max(components)], [intercept + slope * x for x in [0, max(components)]], 'b', linewidth=2)  
    plt.title("y = {} + {}x".format(intercept, slope))
    plt.show()

# %%
