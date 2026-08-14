"""
公众号：阿旭算法与机器学习
专注python与人工智能知识分享，关注可获得更多干货内容哦~
"""

#coding:utf-8
# 全景分割示例
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image = cv2.imread('notebooks/images/picture2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

import sys
sys.path.append("..")
from mobile_encoder.setup_mobile_sam import setup_model
checkpoint = torch.load('weights/mobile_sam.pt',map_location=torch.device('cpu'))
mobile_sam = setup_model()
mobile_sam.load_state_dict(checkpoint,strict=True)

from segment_anything import SamAutomaticMaskGenerator
device = "cpu"
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()