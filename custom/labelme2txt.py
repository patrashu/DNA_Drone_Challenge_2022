import os
import json
import shutil
from glob import glob


CATEGORIES = {
    "person": 0,
    "bicycle": 1,
    "bag": 2,
    "hat": 3
}

def main(subset:str, path: os.PathLike):
    json_paths = glob(path)
    print(json_paths)

    for json_path in json_paths:
        try:
            with open(json_path, 'r') as f:
                annots = json.load(f)

            s = json_path.split('/')
            txt_path = '/'.join(s[:-2]) + f'/{subset}/labels/{s[-1][:-5]}.txt'

            width = annots['imageWidth']
            height = annots['imageHeight']

            f = open(txt_path, 'a')

            for shape in annots['shapes']:
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]

                w, h = x2-x1, y2-y1
                cx = (x1+w//2) / width
                cy = (y1+h//2) / height
                w, h = w/width, h/height
                category_id = CATEGORIES[shape['label']]

                f.write('%s %.3f %.3f %.3f %.3f\n'%(category_id, cx, cy, w, h))

            f.close()

            image_path = '/'.join(s[:-1]) + f'/{s[-1][:-5]}.jpg'
            shutil.copy(image_path, f'{path}/{subset}/images/{s[-1][:-5]}.jpg')

        except:
            print(json_path)
            pass


if __name__ == '__main__':
    json_glob_path = ""
    subset = ""
    main(subset, json_glob_path)

