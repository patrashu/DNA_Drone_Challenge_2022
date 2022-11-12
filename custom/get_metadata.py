from PIL import Image
from PIL.ExifTags import TAGS
# from distutils.util import byte_complie
import re
import ast

'''
Focal Length : 렌즈 초점 길이
ExifImageWidth : 이미지 너비 
ExifImageHeight : 이미지 높이
Exposure Time : 노출 시간
FocalLengthIn35mmFilm : 24
'''

img_name = "ASHuzNAeiV.JPG"

# Read Image
image = Image.open(img_name)
exifdata = image._getexif()

store_dict = dict()

for tag_id in exifdata:
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)

    if isinstance(data, bytes):
        if tag == 'MakerNote':
            for d in re.findall('\[.*?\]', repr(data)):
                try:
                    k, v = d[1:-1].split(":")
                    v1 = f"b'{v}'"
                    v1 = ast.literal_eval(v1)
                    store_dict[k] = v1.decode(encoding='cp949', errors='ignore')
                except:
                    # print(d)
                    pass
        else:
            data = data.decode()

    store_dict[tag] = data

for k,v in store_dict.items():
    print(f"{k:25} : {v}")    
    


