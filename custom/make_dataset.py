import os
import cv2
import json
import shutil
from glob import glob
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm
from torchvision.transforms import Compose, Resize, ToTensor
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## 추가할 객체의 클래스 정의
CATEGORIES = {
    'person': 0,
    'bicycle': 1,
    'bag': 2,
    'hat': 3
}

L_FLAG = False
R_FLAG = False

## 텍스트 파일 내 객체 좌표를 바탕으로 크롭된 객체 이미지 저장
def extract_object_with_bbox():
    cnt = 0
    image_paths = glob('../datasets/type1_m1_dataset/train/images/*.jpg')
    txt_paths = glob('../datasets/type1_m1_dataset/train/labels/*.txt')
    image_paths.sort()
    txt_paths.sort()
    print(len(image_paths), len(txt_paths))

    for image_path, txt_path in zip(image_paths, txt_paths):
        image = Image.open(image_path)
        np_img = np.array(image)
        width, height = image.width, image.height

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line[0] == '0':
                try:
                    line = line[:-2]
                    line = line.split()[1:]
                    cx, cy, w, h = list(map(float, line))
                    w = w * width
                    h = h * height
                    x1 = (cx*width) - w//2
                    y1 = (cy*height) - h//2
                    x2 = x1 + w
                    y2 = y1 + h
                    cropped_image = np_img[int(y1):int(y2), int(x1):int(x2), :]
                    plt.imsave(f'before/person/person{cnt}.jpg', cropped_image)
                    cnt += 1

                except:
                    pass

## labelme를 통해 레이블링 된 객체의 폴리곤 좌표를 바탕으로 크롭된 객체 이미지 저장
def extract_object_with_segmentation():
    root = "../datasets/type1_m1_dataset/add_to_train/images"
    json_paths = glob(f'{root}/*.json')
    i = 0

    print(len(json_paths))
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            json_list = json.load(f)
        
        s = json_path.split('/')
        image_path = '/'.join(s[:-1]) + '/' + json_list['imagePath']
        image = Image.open(image_path)
        shapes = json_list['shapes']
        
        for shape in shapes:
            polygon = shape['points']
            polygon = [(x, y) for x, y in polygon]

            np_img = np.array(image)
            mask_img = Image.new('1', (np_img.shape[1], np_img.shape[0]), 0)
            ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)

            mask = np.array(mask_img)
            new_np_img = np.empty(np_img.shape, dtype='uint8')
            new_np_img = np_img[:, :, :3]
            new_np_img[:, :, 0] = new_np_img[:, :, 0] * mask
            new_np_img[:, :, 1] = new_np_img[:, :, 1] * mask
            new_np_img[:, :, 2] = new_np_img[:, :, 2] * mask

            res_img = Image.fromarray(new_np_img, 'RGB')
            res_img.save(f'./before/person{i}.jpg')
            i += 1


def mouse_event(event, x, y, flags, param):
    global offset_list
    if event == cv2.EVENT_MOUSEMOVE:
        pass
    
    else:
        # if event == cv2.EVENT_MOUSEWHEEL:
        #     offset_list.append((x, y))
        #     L_FLAG = True
        #     R_FLAG = False

        # elif event == cv2.EVENT_RBUTTONDOWN:
        #     offset_list.append((x, y))
        #     R_FLAG = True
        #     L_FLAG = False

        # rand_object = None
        # if L_FLAG:
        #     rand_object = glob('/home/zeroone/Documents/Documentss/yolov7/custom/before/person/*.png')

        # elif R_FLAG:
        #     rand_object = glob('/home/zeroone/Documents/Documentss/yolov7/custom/before/person/*.png')

        if event == cv2.EVENT_RBUTTONDOWN:
            offset_list.append((x, y))

        # 추가할 객체 이미지 전처리 (크기, 방향 조정)
        rand_object = glob('/home/zeroone/Documents/Documentss/yolov7/custom/before/person/*.png')
        object_path = random.choice(rand_object)
        np_object_image = cv2.imread(object_path)
        np_object_image = np_object_image[..., ::-1]

        idx = len(offset_list)-1
        resize_w = np_object_image.shape[1] // 11
        resize_h = np_object_image.shape[0] // 11

        transform = Compose([
            Resize((resize_h, resize_w)),
        ])

        pil_image = Image.fromarray(np_object_image)
        np_object_image = transform(pil_image)
        np_object_image = np.array(np_object_image)


        direction = random.randint(0, 4)
        np_object_image = np.rot90(np_object_image, direction)

        # 클릭한 offset을 기준으로 객체 추가
        x1, y1 = offset_list[idx][0], offset_list[idx][1]
        p_h, p_w = np_object_image.shape[0], np_object_image.shape[1]
        roi = copy_np_img[y1:y1+p_h, x1:x1+p_w, :]
        
        # bitwise 연산
        mask = cv2.cvtColor(np_object_image, cv2.COLOR_BGR2GRAY)
        mask[mask[:]==255] = 0
        mask[mask[:]>0] = 255

        mask_inv = cv2.bitwise_not(mask)
        bg_removed_object = cv2.bitwise_and(np_object_image, np_object_image, mask=mask)
        object_removed_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        dst = cv2.add(bg_removed_object, object_removed_bg)

        copy_np_img[y1:y1+p_h, x1:x1+p_w, :] = dst
        cx, cy = (x1+p_w//2) / width, (y1+p_h//2) / height
        w, h = p_w/width, p_h/height
        
        f.write('%s %.3f %.3f %.3f %.3f\n'%(3, cx, cy, w, h))


if __name__ == '__main__':
    '''
    객체 추가 전 객체 추출
        split_pothole_bbox()
        split_pothole_segmentation()

    가상 객체 추가 알고리즘
        객체가 존재하는 이미지 내에 추가 => flag= 0
        객체가 존재하지 않는 이미지 내에 추가 => flag= 1

    이미지 및 텍스트 파일 경로 
        ~~~/train/images/xxx.jpg
        ~~~/train/labels/xxx.jpg
    '''

    # split_pothole_bbox()
    # split_pothole_segmentation()

    subset = 'added_train'
    raw_image_paths = glob(f'../datasets/type1_m1_dataset/raw/train_images/*.jpg')
    save_root_path = ""
    raw_image_paths.sort()
    
    flag = 0
    if flag == 0: 
        for i, image_path in enumerate(raw_image_paths):
            # print(len(image_paths)-i)
            img = Image.open(image_path)
            np_img = np.array(img)
            width, height = img.width, img.height
            copy_np_img = np_img.copy()

            image_path = image_path.replace('\\', '/')
            s = image_path.split('/')
            txt_path = '/'.join(s[:-2]) + f'/labels/{s[-1][:-4]}.txt'

            with open(txt_path, 'r') as f:
                lines = f.readlines()
            f.close()

            split = txt_path.split('/')
            f = open(f'../{save_root_path}/{subset}/labels/{split[-1]}', 'w')

            for line in lines:
                f.write(line)

            offset_list = []

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 1080, 720)
            cv2.imshow("image", copy_np_img)
            cv2.setMouseCallback('image', mouse_event, copy_np_img)
            cv2.waitKey()

            copy_np_img = cv2.cvtColor(copy_np_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'../{save_root_path}/{subset}/images/{split[-1][:-4]}.jpg', copy_np_img)
            f.close()

            # shutil.move(image_path, '/'.join(s[:-3]) + f'/raw/train_images/{s[-1]}')
            # shutil.move(txt_path, '/'.join(s[:-3]) + f'/raw/train_labels/{split[-1][:-4]}.txt')


    else:
        for i, image_path in enumerate(raw_image_paths):
            # print(len(raw_image_paths)-i)
            img = cv2.imread(image_path)
            np_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height, width = np_img.shape[0], np_img.shape[1]
            copy_np_img = np_img.copy()
            image_path = image_path.replace('\\', '/')
            s = image_path.split('/')

            f = open(f'/{save_root_path}/{subset}/labels/{s[-1][:-4]}.txt', 'w')
            offset_list = []

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 640, 480)
            cv2.imshow("image", img)
            cv2.setMouseCallback('image', mouse_event, copy_np_img)
            cv2.waitKey()
            copy_np_img = cv2.cvtColor(copy_np_img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f'{save_root_path}/{subset}/images/{s[-1]}', copy_np_img)
            f.close()

            # shutil.move(image_path, '/'.join(s[:-3]) + f'/raw/validation_images/{s[-1]}')
        
