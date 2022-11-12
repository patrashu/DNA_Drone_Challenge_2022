from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from tqdm.auto import tqdm
import cv2
import os
from torchvision.transforms import Compose, Resize, ToTensor
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def split_4k_random_extract(crop_h, crop_w, crop_cnt):
    image_paths = glob('../datasets/type1_m1_dataset/train/images/*.jpg')
    txt_paths = glob('../datasets/type1_m1_dataset/train/labels/*.txt')
    print(len(image_paths), len(txt_paths))

    image_paths.sort()
    txt_paths.sort()

    for image_path, txt_path in tqdm(zip(image_paths, txt_paths)):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        img = Image.open(image_path)
        width, height = img.size

        for i in range(len(lines)):
            line = lines[i][:-2].split()[1:]
            cx, cy, w, h = list(map(float, line))

            w, h = w * width, h * height
            cx, cy = cx*width, cy*height
            
            x1 = cx - w//2
            y1 = cy - h//2
            x2 = x1 + w
            y2 = y1 + h

            search_x1 = x2-crop_w+100
            search_y1 = y2-crop_h+100
            search_x2 = x1-100
            search_y2 = y1-100

            crop_x1 = np.random.randint(search_x1, search_x2, crop_cnt)
            crop_y1 = np.random.randint(search_y1, search_y2, crop_cnt)
            np_img = np.array(img)
            
            for i, (xx, yy) in enumerate(zip(crop_x1, crop_y1)):
                try:
                    if i == crop_cnt:
                        break

                    save_img_w = width + 1000*2 
                    save_img_h = height + 1000*2

                    save_img = np.random.randint(0, 1, save_img_w*save_img_h*3)
                    save_img = np.reshape(save_img, (save_img_h, save_img_w, -1))
                    save_img[1000:-1000, 1000:-1000, :] = np_img[:, :, :]
                    save_img = save_img.astype(np.uint8)

                    xx, yy = xx+1000, yy+1000

                    n_save_img = save_img[yy:yy+crop_h, xx:xx+crop_w, :]
                    n_cx = (cx + 1000 - xx) / crop_w
                    n_cy = (cy + 1000 - yy) / crop_h
                    nw = w / crop_w
                    nh = h / crop_h

                    c_img_path = image_path.split('/')
                    c_txt_path = txt_path.split('/')
                    root_path = f'../drone_dataset/cropped_train'
                    save_img_name = f'{root_path}/images/{c_img_path[4][:-4]}_split{i}.jpg'
                    save_txt_name = f'{root_path}/labels/{c_txt_path[4][:-4]}_split{i}.txt'
                    print(n_save_img.shape)
            
                    plt.imsave(save_img_name, n_save_img)
                    f = open(save_txt_name, 'w')
                    ans = '{} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(0, n_cx, n_cy, nw, nh)
                    f.write(ans)
                    f.close()
                except:
                    continue


# 1280 x 1280
def split_4k_image_with_static():
    image_paths = glob('../datasets/type1_m1_dataset/added_train/images/*.jpg')
    image_paths.sort()

    cnt = 0
    for image_path in tqdm(image_paths):
        try:
            split = image_path.split('/')
            txt_path = f'../datasets/type1_m1_dataset/added_train/labels/{split[-1][:-4]}.txt'

            with open(txt_path, 'r') as f:
                lines = f.readlines()

            crop_size = 1280
            x_overlab = 300
            y_overlab = 300
            img = Image.open(image_path)
            width, height = img.width, img.height

            # make cropped images by bruth force
            for j in range(0, height, crop_size-y_overlab):
                for i in range(0, width, crop_size-x_overlab):
                    try:
                        off_x1, off_y1 = i, j
                        off_x2, off_y2 = i+crop_size, j+crop_size

                        if off_x2 > width:
                            off_x1, off_x2 = width - crop_size, width
                        if off_y2 > height:
                            off_y1, off_y2 = height - crop_size, height
                        
                        saved_bbox = []

                        for line in lines:
                            category = line[0]
                            line = line[:-2]
                            line = line.split()[1:]
                            cx, cy, w, h = list(map(float, line))
                            
                            # # load image file and load width and height
                            w = w * width
                            h = h * height
                            x1 = (cx*width) - w//2
                            y1 = (cy*height) - h//2
                            x2 = x1 + w
                            y2 = y1 + h

                            if off_x1 <= x1 <= off_x2 and off_x1 <= x2 <= off_x2:
                                if off_y1 <= y1 <= off_y2 and off_y1 <= y2 <= off_y2:
                                    saved_bbox.append([category, x1, y1, x2, y2])

                        np_img = np.array(img)
                        cropped_np_img = np_img[off_y1:off_y2, off_x1:off_x2, :]

                        ## visualize result of cropping
                        # print(len(saved_bbox))
                        # for c_id, x1, y1, x2, y2 in saved_bbox:
                        #     w, h = x2-x1, y2-y1
                        #     # n_cx = (x1 - off_x1 + w//2) / 1280
                        #     # n_cy = (y1 - off_y1 + h//2) / 1280
                        #     # nw = w / 1280
                        #     # nh = h / 1280

                        #     n_cx = (x1 - off_x1 + w//2)
                        #     n_cy = (y1 - off_y1 + h//2)
                        #     nw = w
                        #     nh = h
                            
                        #     rect = patches.Rectangle(
                        #         (n_cx-nw//2, n_cy-nh//2), nw, nh, fill=False
                        #     )
                        #     ax.add_patch(rect)

                        # plt.imshow(cropped_np_img)
                        # plt.tight_layout()
                        # plt.show()

                        if len(saved_bbox) > 0:
                            c_img_path = image_path.split('/')
                            c_txt_path = txt_path.split('/')
                            root_path = f'../datasets/type1_m1_dataset/cropped_train'
                            save_img_name = f'{root_path}/images/{c_img_path[-1][:-4]}_split{cnt}.jpg'
                            save_txt_name = f'{root_path}/labels/{c_txt_path[-1][:-4]}_split{cnt}.txt'
                            plt.imsave(save_img_name, cropped_np_img)
                            plt.clf()
                            f = open(save_txt_name, 'w')

                            for c_id, x1, y1, x2, y2 in saved_bbox:
                                w, h = x2-x1, y2-y1
                                n_cx = (x1 - off_x1 + w//2) / crop_size
                                n_cy = (y1 - off_y1 + h//2) / crop_size
                                nw = w / crop_size
                                nh = h / crop_size
                                ans = '{} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(c_id, n_cx, n_cy, nw, nh)
                                f.write(ans)

                            cnt += 1
                            f.close()

                    except:
                        print(image_path)

            img.close()

        except:
            print('pass')


if __name__ == '__main__':
    split_4k_image_with_static()
    # split_4k_random_extract(1280, 1280, 5)
