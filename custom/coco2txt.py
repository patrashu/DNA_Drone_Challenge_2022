import json
import os
from tqdm import tqdm

CATEGORIES = {
    '1': 0,
    '2': 1,
    '31': 2,
    '33': 2,
}

def main(subset, root_path):
    with open(f'{root_path}/instances_{subset}.json') as json_f:
        files = json.load(json_f)

    images = files['images']
    annotations = files['annotations']

    image_dict = dict()
    for i, img in enumerate(images):
        image_dict[img['id']] =  i

    if not os.path.exists(f'{root_path}/{subset}/labels'):
        os.mkdir(f'{root_path}/{subset}/labels')
    
    for k in tqdm(range(len(annotations))):
        annot = annotations[k]
        if annot['category_id'] in [1, 2, 31, 33]:
            image = images[image_dict[annot['image_id']]]
            image_id = str(image['file_name'][:-4])

            save_path = f'{root_path}/{subset[:-4]}_label'
            save_name = f'{save_path}/{image_id}.txt'

            width = image['width']
            height = image['height']
            x, y, w, h = annot['bbox']
            category_id = CATEGORIES[str(annot['category_id'])]

            if not os.path.exists(save_name):
                f = open(save_name, 'w')
            else:
                f = open(save_name, 'a')

            cx = (x+w//2) / width
            cy = (y+h//2) / height
            w = w / width
            h = h / height

            ans = '{} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(category_id, cx, cy, w, h)
            f.write(ans)
            f.close()

    json_f.close()


if __name__ == '__main__':
    subset = 'train2017'
    root_path = ""
    main(subset)
