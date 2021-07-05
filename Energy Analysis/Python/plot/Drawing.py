'''
# 
# Description:
#       绘制结果图.
# Author:
#       Xinghua.He
# History:
#       2021.1.12
# 
# 
'''
# %% Import packages.
# System packages.
import os
import sys

# Extend packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
plt.rc('font',family='Times New Roman')
# plt.rc('font', weight='bold') 
import seaborn as sns

# %% Load data.
# Current path.
ROOTPATH = os.path.dirname(os.path.abspath(__file__))
# BP data path.
element_DATAPATH = os.path.join(ROOTPATH, 'result', 'Keras', "550_elements.csv") # Dataframe.
# predict_DATAPATH = os.path.join(ROOTPATH, 'result', 'Keras', "550_predict_sacle.csv") # numpy.
target_DATAPATH = os.path.join(ROOTPATH, 'result', 'Keras', "550_target.csv") # numpy.
# Load data.
element_BP = pd.read_csv(element_DATAPATH)
target_BP = pd.read_csv(target_DATAPATH, names=['Cr_T', 'Mn_T', 'Cu_T'])

# LDNN data path.
element_LDNN_DATAPATH = os.path.join(ROOTPATH, 'result', 'tf', "550_elements.csv") # Dataframe.
predict_LDNN_DATAPATH = os.path.join(ROOTPATH, 'result', 'tf', "550_predictions.csv") # Dataframe.
target_LDNN_DATAPATH = os.path.join(ROOTPATH, 'result', 'tf', "550_target.csv") # numpy.
# Load data.
element_LDNN = pd.read_csv(element_LDNN_DATAPATH)
predict_LDNN = pd.read_csv(predict_LDNN_DATAPATH)
target_LDNN = pd.read_csv(target_LDNN_DATAPATH, names=['Cr_T', 'Mn_T', 'Cu_T'])

assemble_df = pd.concat([element_LDNN, element_BP, target_BP], axis=1)
assemble_df[assemble_df < 0] = 0.00001
assemble_df['number of samples'] = [i+1 for i in range(12)]

# 比较BP与LDNN预测效果.
# %% Cr
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax1, ax2 = ax.ravel()
# Fig 1.
ax1.set_xlabel(..., fontsize=16, weight='bold')
ax1.set_ylabel(..., fontsize=16, weight='bold')
ax1.set_title(..., fontsize=16, weight='bold')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Cr_T'].values, 'o-', color='g', label='Standard')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Cr'].values, 'v-', color='b', label='HDMRN')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Cr_BP'].values, 's-', color='r', label='BP')
ax1.set(title="Comparison of content prediction results (Cr)", xlabel="Samples", ylabel="Content(%)")
ax1.legend(ncol=1, loc="upper center", frameon=True)
# Fig 2.
# bar graphs
ax2.set_xlabel(..., fontsize=16, weight='bold')
ax2.set_ylabel(..., fontsize=16, weight='bold')
ax2.set_title(..., fontsize=16, weight='bold')
width = 0.25
x = assemble_df['number of samples'].values
ax2.bar(x, np.sqrt((assemble_df['Cr'].values - assemble_df['Cr_T'])**2), width, label='HDMRN')
ax2.bar(x + width, np.sqrt((assemble_df['Cr_BP'].values - assemble_df['Cr_T'])**2), width, label='BP')
ax2.set_xticks(x)
ax2.set(title="Prediction error (Cr)", xlabel="Samples", ylabel="Error value")
ax2.legend(ncol=1, loc="upper center", frameon=True)
# plt.show()
plt.savefig("./result/1.png", dpi=2400, bbox_inches='tight')

# %% Mn
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax1, ax2 = ax.ravel()
# Fig 1.
ax1.set_xlabel(..., fontsize=16, weight='bold')
ax1.set_ylabel(..., fontsize=16, weight='bold')
ax1.set_title(..., fontsize=16, weight='bold')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Mn_T'].values, 'o-', color='g', label='Standard')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Mn'].values, 'v-', color='b', label='HDMRN')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Mn_BP'].values, 's-', color='r', label='BP')
ax1.set(title="Comparison of content prediction results (Mn)", xlabel="Samples", ylabel="Content(%)")
ax1.legend(ncol=1, loc="upper left", frameon=True)
# Fig 2.
# bar graphs
ax2.set_xlabel(..., fontsize=16, weight='bold')
ax2.set_ylabel(..., fontsize=16, weight='bold')
ax2.set_title(..., fontsize=16, weight='bold')
width = 0.25
x = assemble_df['number of samples'].values
ax2.bar(x, np.sqrt((assemble_df['Mn'].values - assemble_df['Mn_T'])**2), width, label='HDMRN')
ax2.bar(x + width, np.sqrt((assemble_df['Mn_BP'].values - assemble_df['Mn_T'])**2), width, label='BP')
ax2.set_xticks(x)
ax2.set(title="Prediction error (Mn)", xlabel="Samples", ylabel="Error value")
ax2.legend(ncol=1, loc="upper left", frameon=True)
plt.savefig("./result/2.png", dpi=2400, bbox_inches='tight')

# %% Cu
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax1, ax2 = ax.ravel()
# Fig 1.
ax1.set_xlabel(..., fontsize=16, weight='bold')
ax1.set_ylabel(..., fontsize=16, weight='bold')
ax1.set_title(..., fontsize=16, weight='bold')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Cu_T'].values, 'o-', color='g', label='Standard')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Cu'].values, 'v-', color='b', label='HDMRN')
ax1.plot(assemble_df['number of samples'].values, assemble_df['Cu_BP'].values, 's-', color='r', label='BP')
ax1.set(title="Comparison of content prediction results (Cu)", xlabel="Samples", ylabel="Content(%)")
ax1.legend(ncol=1, loc="upper center", frameon=True)
# Fig 2.
# bar graphs
ax2.set_xlabel(..., fontsize=16, weight='bold')
ax2.set_ylabel(..., fontsize=16, weight='bold')
ax2.set_title(..., fontsize=16, weight='bold')
width = 0.25
x = assemble_df['number of samples'].values
ax2.bar(x, np.sqrt((assemble_df['Cu'].values - assemble_df['Cu_T'])**2), width, label='HDMRN')
ax2.bar(x + width, np.sqrt((assemble_df['Cu_BP'].values - assemble_df['Cu_T'])**2), width, label='BP')
ax2.set_xticks(x)
ax2.set(title="Prediction error (Cu)", xlabel="Samples", ylabel="Error value")
ax2.legend(ncol=1, loc="upper center", frameon=True)
plt.savefig("./result/3.png", dpi=2400, bbox_inches='tight')

# 分析LDNN不同深度的输出.
# %% 
prediction1 = np.array([[float(d) for d in predict_LDNN['prediction1'].values[i].split('[')[1:][0].strip().split(']')[0].strip().split(' ') if d is not ''] for i in range(12)])
prediction2 = np.array([[float(d) for d in predict_LDNN['prediction2'].values[i].split('[')[1:][0].strip().split(']')[0].strip().split(' ') if d is not ''] for i in range(12)])
prediction3 = np.array([[float(d) for d in predict_LDNN['prediction3'].values[i].split('[')[1:][0].strip().split(']')[0].strip().split(' ') if d is not ''] for i in range(12)])

# predictions = np.concatenate([prediction1, prediction2, prediction3], axis=0)
# prediction1 = pd.DataFrame({'Cr':prediction1[:, 0], 'Mn':prediction1[:, 1], 'Cu':prediction1[:, 2]})
prediction1 = pd.DataFrame(prediction1, columns=['Cr', 'Mn', 'Cu'], index=None)
prediction1[prediction1 < 0] = 0.000001
prediction2 = pd.DataFrame(prediction2, columns=['Cr', 'Mn', 'Cu'])
prediction2[prediction2 < 0] = 0.000001
prediction3 = pd.DataFrame(prediction3, columns=['Cr', 'Mn', 'Cu'])
prediction3[prediction3 < 0] = 0.000001

prediction1["layer"] = "Output layer 1"
prediction2["layer"] = "Output layer 2"
prediction3["layer"] = "Output layer 3"

predictions = pd.concat([prediction1, prediction2, prediction3], axis=0)
predictions['number of samples'] = [i+1 for i in range(36)]

# %%
# plt.style.use('ggplot')
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig, ax1 = plt.subplots()
# sns.displot(y='Cr', data=predictions, hue='layer')
# Fig 1. Cr
element_LDNN[element_LDNN < 0] = 0.00001
x = np.array([i+1 for i in range(12)])
y1, y2, y3, y = prediction1['Cr'].values, prediction2['Cr'].values, prediction3['Cr'].values, target_LDNN["Cr_T"].values
width = 0.25 
ax1.bar(x, y1, width, label="output layer 1")
ax1.bar(x + width, y2, width,
        color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], label="output layer 2")
ax1.bar(x + width + width, y3, width,
        color=list(plt.rcParams['axes.prop_cycle'])[4]['color'], label="output layer 3")
ax1.plot(x + width, y, 'o-', color='g', label='Standard')
ax1.plot(x + width, element_LDNN['Cr'].values, 's-', color='b', label='final prediction')
ax1.set_xticks(x + width)
ax1.set_xticklabels([f'{i+1}' for i in range(12)])
ax1.set(title="Cr", xlabel="Samples", ylabel="Content")
ax1.set(ylim=(0, 0.014))
ax1.legend(ncol=1, loc="upper right", frameon=True)

ax1.set_title(label='Cr', fontsize=16, family='Times New Roman', weight='bold')
ax1.set_xlabel(xlabel="Samples", fontsize=16, family='Times New Roman', weight='bold')
ax1.set_ylabel(ylabel="Content(%)", fontsize=16, family='Times New Roman', weight='bold')

plt.savefig("./result/Cr.png", dpi=2400, bbox_inches='tight')

# %% Fig 2. Mn
fig, ax2 = plt.subplots()
y1, y2, y3, y = prediction1['Mn'].values, prediction2['Mn'].values, prediction3['Mn'].values, target_LDNN["Mn_T"].values
width = 0.25 
ax2.bar(x, y1, width, label="output layer 1")
ax2.bar(x + width, y2, width,
        color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], label="output layer 2")
ax2.bar(x + width + width, y3, width,
        color=list(plt.rcParams['axes.prop_cycle'])[4]['color'], label="output layer 3")
ax2.plot(x + width, y, 'o-', color='g', label='Standard')
ax2.plot(x + width, element_LDNN['Mn'].values, 's-', color='b', label='final prediction')
ax2.set_xticks(x + width)
ax2.set_xticklabels([f'{i+1}' for i in range(12)])
ax2.set(title="Mn", xlabel="Samples", ylabel="Content")
ax2.set(ylim=(0, 0.26))
ax2.legend(ncol=1, loc="upper left", frameon=True)

ax2.set_title(label='Mn', fontsize=16, family='Times New Roman', weight='bold')
ax2.set_xlabel(xlabel="Samples", fontsize=16, family='Times New Roman', weight='bold')
ax2.set_ylabel(ylabel="Content(%)", fontsize=16, family='Times New Roman', weight='bold')

plt.savefig("./result/Mn.png", dpi=2400, bbox_inches='tight')

# %%Fig 3. Cu
fig, ax3 = plt.subplots()
y1, y2, y3, y = prediction1['Cu'].values, prediction2['Cu'].values, prediction3['Cu'].values, target_LDNN["Cu_T"].values
width = 0.25 
ax3.bar(x, y1, width, label="output layer 1")
ax3.bar(x + width, y2, width,
        color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], label="output layer 2")
ax3.bar(x + width + width, y3, width,
        color=list(plt.rcParams['axes.prop_cycle'])[4]['color'], label="output layer 3")
ax3.plot(x + width, y, 'o-', color='g', label='Standard')
ax3.plot(x + width, element_LDNN['Cu'].values, 's-', color='b', label='final prediction')
ax3.set_xticks(x + width)
ax3.set_xticklabels([f'{i+1}' for i in range(12)])
ax3.set(title="Cu", xlabel="Samples", ylabel="Content")
ax3.set(ylim=(0, 0.14))
ax3.legend(ncol=1, loc="upper center", frameon=True)

ax3.set_title(label='Cu', fontsize=16, family='Times New Roman', weight='bold')
ax3.set_xlabel(xlabel="Samples", fontsize=16, family='Times New Roman', weight='bold')
ax3.set_ylabel(ylabel="Content(%)", fontsize=16, family='Times New Roman', weight='bold')

plt.savefig("./result/Cu.png", dpi=2400, bbox_inches='tight')

# %%
# plot figure.
f, ax = plt.subplots()
sns.set_color_codes("pastel")
sns.barplot(x="number of samples", y="Cr1", data=predict_df, label="Output Layer 1", color="r")
# % Change bar width.
# columncounts = [5 for i in range(12)]
def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths
# widthbars = normaliseCounts(columncounts,12)
# # Loop over the bars, and adjust the width (and position, to keep the bar centred)
# for bar,newwidth in zip(ax.patches,widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     centre = x+width/2.

#     bar.set_x(centre-newwidth/2.)
#     bar.set_width(newwidth)
# %%
sns.set_color_codes("pastel")
sns.barplot(x="number of samples", y="Cr2", data=predict_df, label="Output Layer 2", color="b")
# % Change bar width.
columncounts = [5 for i in range(12)]
widthbars = normaliseCounts(columncounts,12)
# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)

ax.legend(ncol=2, loc="upper right", frameon=True)
ax.set(ylim=(0,0.01),xlabel="Number of samples", ylabel="Content(%)")
ax.set(title="Cr")
sns.despine(left=True, bottom=True)

# %%

