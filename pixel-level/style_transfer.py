import csv
import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np, toimage
import scipy.misc
import random
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Style Transfer with GFDA')
    parser.add_argument('--src_path', default="/home/liuxinyu/content/datasets/polyps/CVC/CVC_png/", help='Source Dir')
    parser.add_argument('--trg_path', default="/home/liuxinyu/content/datasets/polyps/ETIS/ETIS_png/", help='Target Dir')
    parser.add_argument('--save_path', default="./GFDA_images/", help='Save path')
    parser.add_argument('--anno_file', default="test.csv", help='The csv file for saving correspondance')
    args = parser.parse_args()

    anno_file = args.anno_file
    matcher_dict = {}
    with open(anno_file,'r') as csvfile:
        matches = csv.reader(csvfile)
        # read each line of the csv file
        for match in matches:
            matcher_dict[match[0]] = match[1]

    sourcepath = args.src_path
    targetpath = args.trg_path
    savepath = args.save_path
    RANDOM_TARGET = False
    GUIDED_TARGET = True
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    sourcefiles = os.listdir(sourcepath)
    targetfiles = os.listdir(targetpath)
    sourceimgs = []
    targetimgs = []

    # Only use .png image files
    for srcimg in sourcefiles:
        if srcimg.endswith('.png'):
            sourceimgs.append(srcimg)
        else:
            raise NotImplementedError("Only support .png format images")
    for trgimg in targetfiles:
        if trgimg.endswith('.png'):
            targetimgs.append(trgimg)

    # Start converting
    num_source_img = 0
    for srcimg in sourceimgs:
        num_source_img += 1
        im_src = Image.open(sourcepath + srcimg).convert('RGB')
        if RANDOM_TARGET:
            random_tar = random.choice(targetimgs)
        elif GUIDED_TARGET:
            random_tar = matcher_dict[srcimg]
        im_trg = Image.open(targetpath + random_tar).convert('RGB')

        w, h = im_src.size[0], im_src.size[1]
        im_src = im_src.resize( (w,h), Image.BICUBIC )
        im_trg = im_trg.resize( (w,h), Image.BICUBIC )

        im_src = np.asarray(im_src, np.float32)
        im_trg = np.asarray(im_trg, np.float32)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

        src_in_trg = src_in_trg.transpose((1,2,0))
        print('converting: ', srcimg, 'target: ', random_tar, "("+str(num_source_img)+")")
        toimage(src_in_trg, cmin=0.0, cmax=255.0).save(savepath + srcimg)