from hubconf import *
import torch
import pandas as pd 
from PIL import Image
from tqdm import tqdm

model = custom('./runs/train/stf1-e6e/weights/best.pt') 
model.conf = 0.001 # confidence score
model.iou = 0.55 # IoU score
img_path = '../dataset/' # 이미지 경로

prediction_string = [''for i in range(4871)]
image_id = [f'test/{i:04}.jpg' for i in range(4871)]

for i in tqdm(range(len(image_id))):
    img = Image.open(img_path+image_id[i])
    preds = model(img,size=1024,augment=True)
    for data in preds.pandas().xyxy[0].values:
        x1,y1,x2,y2,conf,cls,label = data
        prediction_string[i] += f'{cls} {conf} {x1} {y1} {x2} {y2} ' # 형식에 맞춰서 작성
        
submission = pd.DataFrame()
submission['PredictionString'] = prediction_string
submission['image_id'] = image_id
submission.to_csv("./output/stf1_tta.csv", index=False)