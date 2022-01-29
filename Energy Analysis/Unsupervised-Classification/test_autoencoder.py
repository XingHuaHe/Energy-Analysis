

from array import array
import os
import argparse
from typing import List
from matplotlib.pyplot import imshow
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from models.autoencoder import AutoEncoder_512, AutoEncoder_32
from utils.energyDataset import EnergyDataset
from torch.utils.data import DataLoader


def get_images(img_path: str) -> List:
    """获取指定路径下的所有图片"""

    import os
    
    if not os.path.exists(img_path):
        raise f"{img_path} is not exist"

    results = []
    labels = []

    dirs = os.listdir(img_path)
    for d in dirs:
        if d == "alloy":
            # alloy : 0 
            alloy_path = os.path.join(img_path, "alloy")
            filenames = os.listdir(alloy_path)
            for filename in filenames:
                results.append(os.path.join(alloy_path, filename))
                labels.append(0)
        elif d == "soil":
            # soil : 1
            soil_path = os.path.join(img_path, "soil")
            filenames = os.listdir(soil_path)
            for filename in filenames:
                results.append(os.path.join(soil_path, filename))
                labels.append(1)
        elif d == "com-alloy":
                # common-alloy : 2
                com_alloy_path = os.path.join(img_path, "com-alloy")
                filenames = os.listdir(com_alloy_path)
                for filename in filenames:
                    results.append(os.path.join(com_alloy_path, filename))
                    labels.append(2)

    return results, labels


def kmean_classify(data: array) -> None:


    kmeans = KMeans(n_clusters=3, random_state=None, max_iter=10)
    kmeans.fit(data)
    
    print(kmeans.labels_)
    print(len(kmeans.labels_))


def test_autoencoder(args: argparse.ArgumentParser):
    """
        测试自编码器
    """

    pass


def test_encoder(args: argparse.ArgumentParser):
    """
        测试编码器
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        # transforms.Resize((args.image_size, args.image_size,)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (0.5, 0.5, 0.5))
    ])

    # model
    if args.model_name == 'autoencoder_512':
        model = AutoEncoder_512(image_size=args.image_size, batch_size=args.batch_size, trainable=args.trainable)
    elif args.model_name == 'autoencoder_32':
        model = AutoEncoder_32(image_size=args.image_size, batch_size=args.batch_size, trainable=args.trainable)
    model.to(device)

    if args.weights != '':
        try:
            state_dict = torch.load(args.weights, map_location=device)
            model.load_state_dict(state_dict=state_dict)
        except Exception as e:
            print(e)
    else:
        raise "weights cann't be None !"

    imgs, labels = get_images(args.images_path)

    features = {'feature': [], 'label': []}

    for i in range(len(imgs)):

        img0 = Image.open(imgs[i]).convert("RGB")

        img = test_transform(img0)

        feature = model(torch.unsqueeze(img.to(device), 0))

        features['feature'].append(feature.cpu().numpy()[0])
        features['label'].append(labels[i])

    # df = pd.DataFrame(features)
    # df.to_csv(os.path.join(args.outputs, 'features.csv'))
    a = np.array(features['feature'])
    b = np.array(features['label'])
    b = np.reshape(b, (b.shape[0], 1))
    c = np.concatenate((a, b), axis=1)
    np.save(os.path.join(args.outputs, 'features'), c)
    
    # kmeans classification.
    kmean_classify(np.array(features['feature']))
    print(features['label'])
    print(len(features['label']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="what you want to do")
    parser.add_argument('--images-path', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/datas/images", help="images directory")
    parser.add_argument('--model-name', type=str, default='autoencoder_32', help="model name")
    parser.add_argument('--weights', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/weights/autoencoder_32/autoencoder_32_200_ck.pt", help="pre-training model")
    parser.add_argument('--batch-size', type=int, default=1, help="Traing batch size")
    parser.add_argument('--image-size', type=int, default=32, help="model input images size")
    parser.add_argument('--trainable', type=bool, default=False, help="True for traing or False for evel")
    parser.add_argument('--outputs', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/outputs/test", help="output directory")
    parser.add_argument('--feature-path', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/outputs/test/features.csv", help='feature file')
    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    with torch.no_grad():
        if args.trainable:
            test_autoencoder(args)
        else:
            test_encoder(args)