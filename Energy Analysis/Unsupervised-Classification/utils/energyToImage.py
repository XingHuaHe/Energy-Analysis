import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd


def extract_datas(filepath):
    filenames = os.listdir(filepath)
    alloyDatas = []
    alloyId = []
    for filename in filenames:
        if filename.split('.')[-1].lower() == 'xls':
            # labels = pd.read_excel(os.path.join(filepath, filename))
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

    return alloyDatas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/datas/dat/alloy", help="data source")
    parser.add_argument('--target', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/datas/images/alloy", help="images directory")
    parser.add_argument('--name', type=str, default='alloy', help="data class name")
    args = parser.parse_args()

    # Getting energy data.
    alloyDatas = extract_datas(args.source)

    for i in range(len(alloyDatas)):
        alloyData = alloyDatas[i]
        # plt.plot([i for i in range(2048)], alloyData)
        dim = len(alloyData)
        img = np.zeros((3, dim, dim))
        alloyData = np.array([alloyData])
        alloyData_T = alloyData.transpose()
        arry = alloyData_T.dot(alloyData)
        arry[arry < 10.0] = 10

        arry = np.log10(arry)
        img[0, :, :] = arry
        img[1, :, :] = arry
        img[2, :, :] = arry
        min_value = np.min(arry)
        max_value = np.max(arry)
        img = img / max_value
        img = np.uint8(img * 255)
        img = np.transpose(img, (1, 2, 0))

        img = Image.fromarray(img)
        img = img.resize((32, 32))
        img.save(f"{args.target}/{args.name}-{i}.jpg")
