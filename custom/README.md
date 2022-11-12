# Dataset 구축

## 1. 웹 크롤링
- [크롬 드라이버 다운(106.xxx)](https://chromedriver.chromium.org/downloads)
- python==3.6
- 수집하고자 하는 키워드를 keywords 변수 안에 입력 후 실행 

```python
python crawl.py
```

## 2. 생성적 적대 네트워크를 이용한 (날씨/계절) 데이터셋 구축



## 3. 가상 객체 추가 알고리즘
- 추가 알고리즘 실행 이전 추가할 객체 추출
- 이미지 경로 및 저장 경로 지정
- 추가할 이미지 내 객체 유무에 따라 파일 실행
- 오른쪽 마우스 클릭 이벤트를 통해 가상 객체 추가

```python
python make_dataset.py
```
- 가상 객체 추가 알고리즘 결과


## 4. 4k 이미지 분할 알고리즘
- 슬라이딩 윈도우 방식으로 진행
- crop_size와 nrows, ncols 지정 후 실행
    - Grid를 기준으로 객체 존재 시에 객체 이미지 저장 (default)
    - ![grid](https://user-images.githubusercontent.com/78347296/201474171-5ee8c53e-250b-49fb-bb00-03b83bbc55f0.gif)
    - 객체를 기준으로 일정 구간 내 랜덤 이미지 저장, 초과하는 영역은 latterbox로 대체 후 이미지 저장


<hr>

### Yolo 학습을 위한 txt 변환 코드
- labelme를 활용하여 레이블링이 진행된 이미지를 Yolo 학습 형태에 맞게 변환하는 코드 구현
- COCO_Dataset에서 원하는 Class만 추출하여 학습에 사용하는 코드 구현

```python 
python labelme2txt.py
python coco2txt.py
```

### matplotlib을 활용한 랜덤 이미지 시각화
```python
python vis.py
```

### 디렉토리 이동, 삭제 및 중복 이미지 추출
```python
python utils.py
```

### 이미지 내 메타데이터 추출
```python
python get_metadata.py
```