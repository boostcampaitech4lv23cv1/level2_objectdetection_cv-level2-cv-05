# 재활용 품목 분류를 위한 Object Detection


## 🎬 프로젝트 개요
<img width="819" alt="스크린샷 2023-01-24 오전 12 03 30" src="https://user-images.githubusercontent.com/70750888/214073463-4b29faf1-e2ce-4ab9-81e3-7f17556acf11.png">

- 분리수거는 환경부담을 줄일 수 있는 방법 중 하나이므로 사진에서 쓰레기를 Detection 하는 모델을 만들어 일반 쓰레기, 플라스틱, 종이 등 10종류의 쓰레기를 분리할 수 있는 모델 설계



## 📚 데이터
- 전체 이미지 개수 : 9754장
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)


## 🌳 EDA
![스크린샷 2023-01-24 오전 12 17 01](https://user-images.githubusercontent.com/70750888/214076685-8e19f421-7ef7-4d9d-b460-60d5945c8ed3.png)

- 이미지 내 bbox의 개수, size, aspect ratio, class 분포, 중심점 좌표 위치, 통계치 등을 분석
- 이미지 별 bbox 개수 및 크기 상이, class 불균형을 인지
- image data를 bbox, class와 함께 시각화함으로써 불필요한 bbox 를 인식
- 이미지들의 평균값, 중간값을 이용하여 Anchor box 비율 조정
- EDA를 통한 데이터 기반 의사결정 

## 🛠 Data Preprocessing
### 개별 실험 
- Faster RCNN baseline/ Adam optimizer/ lr 1e-4
|**Epoch**|**1x**|**2x**|
|------|---|---|
|No augmentation|0.422|0.395|
|RandomRotate90|0.437||
|RandomResizedCrop|0.438||
|ClAHE|0.429||
|GaussNoise|0.438||
|Mosaic|0.401|0.445|
|Mix-up|0.316|0.358|

- RandomRotate90, RandomResizedCrop, CLAHE, GaussNoise는 첫번째 실험에서 성능향상을 보여서 후보군에 포함
- Mosaic과 Mixup은 epcoh을 늘리면 성능향상으로 이어질 수도 있다는 가정 하에 2x로 재실험한 결과 Mosaic은 성능향상을 보여 후보군에 포함, mixup은 최종탈락

### 조합 실험 
- Faster RCNN / backbone Resnext / Adam optimizer / lr1e-4
|**Augmentation 조합**|**Best validation mAP 50**|
|------|---|
|All|0.512|
|-mixup|0.535|
|- mixup, GaussNoise|0.538|
|- mixup, GaussNoise, RBC|0.536|
|- mixup, GaussNoise + centercrop, randomresizedcrop|0.527|
|- mixup, GaussNoise, RBC, Hue|0.527|
|- mixup, GaussNoise + Oneof대신 flip/rotate 모두 적용|0.534|
|- mixup, GaussNoise + cutout parameter tune|**0.545**|
|- mixup, aussNoise + Oneof 대신 flip 모두 적용|0.531|
|- mixup, GaussNoise, cutout|0.541|

- 개별 실험에서 성능이 확실히 입증되지는 않은 기법들을 빼가면서 성능 실험을 진행
- 시각화를 통해, 이미지를 직접 눈으로 확인해가며 파라미터 값 설정
- Cutout의 경우, 작은 size의 hole을 더 많이 생성했을 때 성능 향상
- Multi scale 의 경우, 512 ~ 1024까지 64 간격의 scale로 학습 시켰을 때 성능 향상
- Mixup과 Gaussian noise를 제외하고 나머지 모든 augmentation들을 사용했을 때, 가장 성능이 좋아, 최종 모델에 적용



## ⚜️ Model design

- State-of-the-art를 이용하여 모델 선택
- 1-stage모델로 yolov5, v7을 사용
- 2-stage모델로 mmdetection을 이용하여 사용

|**Model**|**Schedulers**|**Optimizers**|
|------|---|---|
|Cascade Swin_base|Step|SGD|
|Cascade Swin_Large|Cosine Annealing|AdamW|
|Cascade Swin_Large(Pseudo-labeling)|Cosine Annealing|AdamW|
|UniverseNet|Cosine Annealing|AdamW|
|Yolo V7|Linear|SGD|
|Yolo V7(Pseudo-labeling)|Linear|SGD|
|Yolo V5|Linear|SGD|
|Faster_RCNN Resnext|Step|Adam|



![스크린샷 2023-01-24 오전 12 42 21](https://user-images.githubusercontent.com/70750888/214082627-c1f15e97-17e9-4562-954b-5a167306ea7b.png)
- confusion matrix/ yolov7

## 💎 TTA & Ensemble

TTA
- Multiscale : 512-1024 64간격으로 진행
- Horizontal Flip

Ensemble (Weighted-Boxes-Fusion를 이용)

- Cascade Swin_large, Cascade Swin_base, Yolo V7 ensemble를 이용
- IoU threshold : 0.55
- Cascade Swin_Large가 가장 좋은 성능을 냈기에 가중치 부여

## 🏆 Results


||**Public Score**|**Private Score**|
|------|---|---|
|k fold Cascade Swin_Large(weight : 3) + k fold YOLOv7 + Cascade Swin_base|0.7127(2등)|**Private : 0.6983 (최종 2등)**|
|k fold Cascade Swin_Large(weight : 4) + Cascade Swin_base + UniverseNet + k fold YoloV7 +  YoloV5 + k fold Faster RCNN Resnext|0.7125|0.6981|
