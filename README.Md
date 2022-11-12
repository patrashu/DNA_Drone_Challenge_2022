# DNA+ drone challenge 
## [지정공모 2. 실시간 AI 서비스 챌린지](http://challenge-dnadrone.com/?page_id=600)
### Mission 1 도로 위 결함을 검출하라


<br>

#### 실험 환경
- Intel® Xeon® Gold 5218R(2.10GHz)
- NVIDIA RTX A4000(16GB) * 2

<br>

#### 개발 환경 구축
```bash
conda create -n yolov7 python=3.8
conda activate yolov7
pip install -r requirement.txt
```

<br>

#### [데이터셋 구축 전략](https://github.com/patrashu/DNA_Drone_Challenge_2022/tree/master/custom)

<br>

#### 학습 및 추론 weight 다운로드
- [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt)
- [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) 
- [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

<br>

#### 학습
```bash
python train.py --workers 12 --device 0,1 --batch-size 8 --data data/custom_type2_drone.yaml --img 1280 1280 --cfg cfg/training/yolov7-type2-drone.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
```

<br>

#### 추론
```bash
python detect_patch.py --weights yolov7.pt --conf 0.25 --img-size 1280 --source datasets --project results
```

<br>

#### [결과 이미지]((https://github.com/patrashu/DNA_Drone_Challenge_2022/tree/master/results)