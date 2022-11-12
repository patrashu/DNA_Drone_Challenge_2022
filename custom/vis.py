import os
import random
from glob import glob

from typing import List
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def vis_3x3_txt(path: List[os.PathLike]):
    '''
    img_path => ~~~/images/xxx.jpg
    label_path => ~~~/labels/xxx.txt
    '''

    ncols, nrows = 3, 3

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, dpi=200)
    image_paths = glob(path)
    j = 0

    for i in range(ncols*nrows):
        image_path = random.choice(image_paths)
        s = image_path.split('/')
        root = '/'.join(s[:-2])
        txt_path = f'{root}/labels/{s[-1][:-4]}.txt'

        img = Image.open(image_path)

        with open(txt_path, 'r') as f:
            txt_info = f.readlines()

        width, height = img.size

        for line in txt_info:
            c, cx, cy, w, h = line.strip().split()
            w = float(w) * width
            h = float(h) * height

            x = float(cx)*width - w // 2
            y = float(cy)*height - h // 2

            rect = patches.Rectangle(
                (x, y), w, h, fill=False, linewidth=0.3, color='red'
            )

            axes[j][i%ncols].add_patch(rect)
            axes[j][i%ncols].axis(False)
        
        if (i+1)%ncols == 0:
            j += 1

        axes.flat[i].imshow(img)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_glob_paths = ""
    vis_3x3_txt(img_glob_paths)



