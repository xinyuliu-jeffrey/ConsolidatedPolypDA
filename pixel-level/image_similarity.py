import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import math

import csv
import cv2
import torch
import random
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

import argparse


def norm_dataset(filepath):
  pathDir = os.listdir(filepath)
  R = 0
  G = 0
  B = 0
  R_tt = 0
  G_tt = 0
  B_tt = 0
  total_pixel = 0
  for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imageio.imread(os.path.join(filepath, filename))
    R = R+np.sum(img[:,:,0])
    G = G+np.sum(img[:,:,1])
    B = B+np.sum(img[:,:,2])
    size1 = img.shape[0]
    size2 = img.shape[1]
    total_pixel = total_pixel + size1*size2

  R_mean = R/total_pixel
  G_mean = G/total_pixel
  B_mean = B/total_pixel

  total_pixel = 0
  for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imageio.imread(os.path.join(filepath, filename))
    size1 = img.shape[0]
    size2 = img.shape[1]
    total_pixel = total_pixel + size1*size2
    R_tt = R_tt + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_tt = G_tt + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_tt = B_tt + np.sum((img[:, :, 2] - B_mean) ** 2)
  R_std = math.sqrt(R_tt / total_pixel)
  G_std = math.sqrt(G_tt / total_pixel)
  B_std = math.sqrt(B_tt / total_pixel)

  dataset_mean = [R_mean/255, G_mean/255, B_mean/255]
  dataset_std = [R_std/255, G_std/255, B_std/255]
  return dataset_mean, dataset_std


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)
  
def weighter(num_imgs):
    weighted = []
    for i in range(num_imgs):
        y = normal_distribution(i/98, 0, 1) # i/65.333 is used for restrict to [-3, 3]
        weighted.append(y)
    weighted = np.array(weighted)
    weighted = weighted/np.sum(weighted) # make the 'weighted' to sum=1
    return weighted

def matched_trg_selector(pair_dict, rule):
    '''
        Select the matched target image according to the rule.
        pair_dict: a dict with keys: target img names. values: cos similarity
        rule: one select rule in ['min', 'max', 'rand']
    '''
    if rule == 'min':
        matched_trg = sorted(pair_dict.items(), key = lambda item:item[1])[0][0]
    elif rule == 'max':
        matched_trg = sorted(pair_dict.items(), key = lambda item:item[1])[-1][0]
    elif rule == 'rand':
        matched_trg = random.sample(pair_dict.keys(), 1)[0]
    elif rule == 'gauss':
        trgs = sorted(pair_dict.items(), key = lambda item:item[1])
        sim_val_sorted = sorted(pair_dict.values())
        float_sim_val_sorted = []
        for i in sim_val_sorted:
            j = i.float()
            float_sim_val_sorted.append(j)
        # print(sim_val_sorted)
        weighted = weighter(196)
        rand_val = np.random.choice(sim_val_sorted, 1, p=weighted)
        rand_val = torch.tensor(rand_val.astype(float)).cuda()
        index = sim_val_sorted.index(rand_val)
        # print(index)
        matched_trg = trgs[index][0]
    return matched_trg


def trg_vecs(trg_path, model):
    trg_dict = {}
    model = model.cuda()
    model.eval()
    for trg_img in os.listdir(trg_path):
        with torch.no_grad():
            if trg_img.endswith(".png"):
                img = Image.open(trg_path + trg_img)
                img_t = transform_trg(img).unsqueeze(0).cuda()
                trg_out = model(img_t)
                trg_dict[trg_img] = trg_out
    return trg_dict

def get_pairs(source_path, trg_dict, model, anno_file, gauss_sample = False):
  model = model.cuda()
  model.eval()
  f = open(anno_file,'w',encoding='utf-8')
  num_of_img = 0
  for src_img in os.listdir(source_path):
    num_of_img += 1
    if src_img.endswith(source_img_format):
      with torch.no_grad():
        img = Image.open(source_path + src_img)
        img_t = transform_src(img).unsqueeze(0).cuda()
        src_out = model(img_t)
        pair_dict = {}
        for trg_img in trg_dict:
          pair_dict[trg_img] = cos(src_out, trg_dict[trg_img])
        if gauss_sample:
          rule = 'gauss'
          matched_trg = matched_trg_selector(pair_dict, rule)
        else:
          if num_of_img <= 153:
            rule = 'min'
            matched_trg = matched_trg_selector(pair_dict, rule)
          elif num_of_img > 759:
            rule = 'max'
            matched_trg = matched_trg_selector(pair_dict, rule)
          else:
            rule = 'rand'
            matched_trg = matched_trg_selector(pair_dict, rule)
      print(src_img + " is matched to: " + matched_trg, " rule: ", rule)
      csv_writer = csv.writer(f)
      csv_writer.writerow([src_img, matched_trg])
  f.close()
  return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Image Similarity')
    parser.add_argument('--src_path', default="./CVC_png/", help='Source Dir')
    parser.add_argument('--trg_path', default="./ETIS_png/", help='Target Dir')
    parser.add_argument('--anno_file', default="test.csv", help='The csv file for saving correspondance')
    args = parser.parse_args()

    source_path = args.src_path
    trg_path = args.trg_path
    anno_file = args.anno_file
    source_img_format = ".png"


    model = models.resnet50(pretrained=True)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    src_dataset_mean, src_dataset_std = norm_dataset(source_path)
    print("==> src dataset mean: {}, src dataset std: {}".format(src_dataset_mean, src_dataset_std))
    trg_dataset_mean, trg_dataset_std = norm_dataset(trg_path)
    print("==> tgt dataset mean: {}, tgt dataset std: {}".format(trg_dataset_mean, trg_dataset_std))
    transform_src = transforms.Compose([            
        transforms.Resize((224, 224), Image.BICUBIC),                    
        transforms.ToTensor(),                     
        transforms.Normalize(
        mean = src_dataset_mean,
        std = src_dataset_std
        )])
    transform_trg = transforms.Compose([            
        transforms.Resize((224, 224), Image.BICUBIC),                    
        transforms.ToTensor(),                     
        transforms.Normalize(
        mean = trg_dataset_mean,
        std = trg_dataset_std
        )])

    model.avgpool = Identity()
    model.fc = Identity()

    trg_dict = trg_vecs(trg_path, model)
    print("TOTAL TARGET IMAGES:", len(trg_dict))

    get_pairs(source_path, trg_dict, model, anno_file, gauss_sample=True)