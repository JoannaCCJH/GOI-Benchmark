#
import os, sys
import cv2
import torch
from torchvision import transforms

import matplotlib
import numpy as np

def calculate_iou(label, pred):
    """
    Calculates Intersection over Union (IoU) as IoU = |Intersection| / |Union| for binary masks.

    Args:
        label (torch.Tensor): Binary ground truth tensor of shape (H, W, 1)
        pred (torch.Tensor): Binary predicted tensor of same shape as label
    """

    pred_inds = pred == 1
    label_inds = label == 1
    intersection = torch.logical_and(pred_inds, label_inds).sum()
    union = torch.logical_or(pred_inds, label_inds).sum()
    if union == 0:
        iou = float('nan')  # 避免除以零
    else:
        iou = float(intersection) / float(max(union, 1))

    return iou

def calculate_mean_pixel_accuracy(true_labels, predicted_labels):
    """
    Computes Mean Pixel Accuracy (mPA) by calculating the average per-class pixel classification accuracy.

    Args:
        true_labels (torch.Tensor): Ground truth tensor of shape (H, W, 1)
        predicted_labels (torch.Tensor): Predicted labels tensor of same shape as true_labels
    """
    
    assert true_labels.shape == predicted_labels.shape

    # Calculate per-class pixel accuracies
    accuracy_class_1 = torch.sum((predicted_labels == 1) & (true_labels == 1)).float() / torch.sum(true_labels == 1).float()
    accuracy_class_0 = torch.sum((predicted_labels == 0) & (true_labels == 0)).float() / torch.sum(true_labels == 0).float()

    accuracy_class_1 = accuracy_class_1 if torch.sum(true_labels == 1) > 0 else torch.tensor(0.)
    accuracy_class_0 = accuracy_class_0 if torch.sum(true_labels == 0) > 0 else torch.tensor(0.)

    # Compute mean pixel accuracy
    mPA = (accuracy_class_1 + accuracy_class_0) / 2
    return mPA
