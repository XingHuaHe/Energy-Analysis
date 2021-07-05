# %%
# System packages.
import os
import sys
# Externed packages.
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F
# User packages.
from train import Energy
# %%
path_checkpoint = "./checkpoints/checkpoint_100_epoch.pkl"
checkpoint = torch.load(path_checkpoint)

model = Energy(num_class=3)
model.load_state_dict(checkpoint['model_stact_dict'])
model.eval

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get("relu3").register_forward_hook(hook_feature)
# %%
# get the softmax weight
params = list(model.parameters())
# %%
weight_softmax = np.squeeze(params[-2].data.numpy())
# %%
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
# %%
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# %%
img_pil = Image.open('./test3.jpg')
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = model(img_variable)
# %%
# download the imagenet category list
classes = {0:"普通合金", 1:"轻合金", 2:"土壤"}

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()
# %%
# output the prediction
for i in range(0, 3):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
# %%
# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
# %%
# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('test3.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM3.jpg', result)

# %%
