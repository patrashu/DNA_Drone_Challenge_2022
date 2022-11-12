import argparse
import time
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import matplotlib
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.dataset_patch import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

COLORS = [(matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int) for x in np.linspace(0, 1, 2, endpoint=False)]
COLORS = [tuple([cx.item() for cx in c]) for c in COLORS]

def nms(boxes, probs, labels, overlap_thresh):
    if not boxes.size:
        return np.empty((0,), dtype=int)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = [] 
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(probs)
    while idxs.size:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int"), probs[pick], labels[pick]


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, w_over, h_over, overlap_thresh, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.w_over, opt.h_over, opt.overlap_thresh, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, w_over=w_over, h_over=h_over)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap, oripatches, patches, starts in dataset:
        img_patches = []
        for img in patches:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            img_patches.append(img)

        # Inference
        t1 = time_synchronized()
        # pred = model(patches)
        preds = []
        for img in img_patches:
            # print(img.shape)
            pred = model(img, augment=opt.augment)[0]
            preds.append(pred)
        t2 = time_synchronized()

        # Apply NMS
        # break
        result_from_model = []
        for pred in preds:
            result = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            result_from_model.append(result)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)    
        draw = im0


        boxes = []
        scores = []
        labels = []
        for patch, coords in zip(result_from_model, starts):
            for res in patch:
                res = res.detach().cpu().numpy()
                if res is not None:
                    res[:,0] += coords[0]
                    res[:,1] += coords[1]
                    res[:,2] += coords[0]
                    res[:,3] += coords[1]
                    for box in res:
                        boxes.append(box[:4])
                        scores.append(box[4])
                        labels.append(box[5])

        print(len(scores))
        if len(boxes) != 0:
            boxes = np.asarray(boxes)
            scores = np.asarray(scores)
            labels = np.asarray(labels)

            boxes, scores, labels = nms(boxes, scores, labels, overlap_thresh=overlap_thresh)
            for box, score, label in zip(boxes, scores, labels):

                b = box.astype(int)
                l = int(label)
                color = COLORS[l]
                cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, 3, cv2.LINE_AA)
                
                label = f'{names[int(l)]} {score:.2f}'
                font = ImageFont.load_default()
                draw_pil = Image.fromarray(draw)
                draw_pil_d = ImageDraw.Draw(draw_pil)
                width, height = draw_pil_d.textsize(label, font)
                draw_pil_d.rectangle(((b[0], b[1] - height), b[0] + width, b[1]), fill=color)
                draw_pil_d.text((b[0], b[1] - height),label,font=font,fill=(0, 0, 0),)
                draw = np.array(draw_pil)

        cv2.imwrite(save_path, draw)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='type1_best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--w-over', type=int, default=100, help='overlay size (pixels)')
    parser.add_argument('--h-over', type=int, default=50, help='overlay size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--overlap-thresh', type=float, default=0.01, help='overlap confidences')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
