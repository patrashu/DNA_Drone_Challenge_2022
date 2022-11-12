import os
import cv2
import shutil

import numpy as np
from glob import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# 이미지와 텍스트 파일 디렉토리 이동
def move(path: os.PathLike, move_path: os.PathLike):
    image_paths = glob(path)
    print(len(image_paths))

    for i, image_path in enumerate(image_paths):
        try:
            image_path = image_path.replace('\\', '/')
            s = image_path.split('/')
            root_path = '/'.join(s[:-2])
            txt_path = f'{root_path}/labels/{s[-1][:-4]}.txt'

            shutil.copy(txt_path, f'{move_path}/labels/{s[-1][:-4]}.txt')
            shutil.copy(image_path, f'{move_path}/images/{s[-1]}')

        except:
            print(image_path)
        
# 웹 크롤링한 이미지 중 중복 이미지 제거
def check_duplicate_image(path: os.PathLike)):
    image_paths = sorted(glob(path))
    hists = []

    for i, img in tqdm(enumerate(image_paths), desc='creating histogram list') :
        try: 
            img_ = cv2.imread(img)

            hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])

            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hists.append((img, hist))

        except TypeError:
            print(f'{img}: damaged image')
            os.remove(img)
            print('deleted.')

        except cv2.error:
            print(img_.shape)
            

    methods = {
        'CORREL': cv2.HISTCMP_CORREL,
    }

    print('비교 시작')
    result = []

    for _, (name, flag) in enumerate(methods.items()):
        print('mode: %-10s'%name, end='\n')
        x = 0
        for img in image_paths:
            img_ = cv2.imread(img)
            hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            for i in range(x, len(hists)):
                img_name, hist_ = hists[i]
                ret = cv2.compareHist(hist, hist_, flag)
                
                if flag == cv2.HISTCMP_INTERSECT:
                    ret = ret/np.sum(hist)
                
                if ret == 1.0 and img != img_name:
                    print(f'img1: {img} | img2: {img_name} | value: {ret}', end='\n')
                    result.append(img_name)
            x += 1
            
        print()
    
    result = set(result)
    print(len(result))

    for res in result:
        os.remove(res)

    print(result)

# 이미지와 텍스트파일 삭제
def del_file(path: os.PathLike):
    image_paths = glob(path)

    for i, image_path in enumerate(image_paths):
        try:
            image_path = image_path.replace('\\', '/')
            s = image_path.split('/')
            root_path = '/'.join(s[:-2])
            txt_path = f'{root_path}/labels/{s[-1][:-4]}.txt'

            os.remove(image_path)
            os.remove(txt_path)

        except:
            print(image_path)


if __name__ == '__main__':
    '''
    이미지 및 텍스트 파일 경로 세팅
        ~~~/train/images/xxx.jpg
        ~~~/train/labels/xxx.jpg
    '''
    
    img_glob_path = '../datasets/type1_m1_dataset/add_to_val/images/*.jpg'
    move_path = ""
    pass
    # del_file(img_glob_path)
    # move(img_glob_path, move_path)
    # check_duplicate_image(img_glob_path)
