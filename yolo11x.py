import os
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def main():
  
  print("-" * 30)
  print("Torch:", torch.version)
  print("GPU is available:", torch.cuda.is_available())
  if torch.cuda.is_available():
  print("GPU Name:", torch.cuda.get_device_name(0))
  print("-" * 30)
  
  
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  PROJECT_DIR = os.path.join(BASE_DIR, "miet_results") 
  os.makedirs(PROJECT_DIR, exist_ok=True)
  
  
  print("Скачивание датасета")
  dataset_path = r"F:\call"
  print("Dataset path:", dataset_path)
  
  
  images_dir = os.path.join(dataset_path, 'yolo_dataset', 'yolo_dataset', 'train', 'images')
  if not os.path.exists(images_dir):
      images_dir = os.path.join(dataset_path, 'yolo_dataset', 'train', 'images')
  
  
  all_images = [
      os.path.join(images_dir, f) for f in os.listdir(images_dir)
      if f.lower().endswith(('.png','.jpg','.jpeg'))
  ]
  train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
  
  train_txt_path = os.path.join(PROJECT_DIR, "train.txt")
  val_txt_path = os.path.join(PROJECT_DIR, "val.txt")
  
  with open(train_txt_path, "w") as f: f.write("\n".join(train_imgs))
  with open(val_txt_path, "w") as f: f.write("\n".join(val_imgs))
  
  
  yaml_path = os.path.join(PROJECT_DIR, "data.yaml")
  data_yaml = {
      "train": train_txt_path,
      "val": val_txt_path,
      "nc": 2,
      "names": ["visitor", "staff"]
  }
  with open(yaml_path, "w") as f: yaml.dump(data_yaml, f)
  
  
  RUN_DIR = os.path.join(PROJECT_DIR, "runs")
  last_weights = os.path.join(RUN_DIR, "lab2_model_v2", "weights", "last.pt") 
  
  if os.path.exists(last_weights):
      model = YOLO(last_weights)
      results = model.train(resume=True)
  else:
      model = YOLO("yolo11x.pt") 
      results = model.train(
          data=yaml_path,
          epochs=150,
          imgsz=960,             
          project=RUN_DIR,
          name="lab2_model_v2",
          batch=4,               
          patience=20,
          optimizer="auto",
          cos_lr=True,
          mosaic=1.0,
          close_mosaic=15,
          erasing=0.4,
          mixup=0.15,            
          copy_paste=0.1,        
          degrees=5.0,
          translate=0.1,
          scale=0.4,
          fliplr=0.5,
          save_period=5,
          workers=4
      )
  print("Финальный инференс")
  best_model_path = os.path.join(RUN_DIR, "lab2_model_v2", "weights", "best.pt")
  model = YOLO(best_model_path)
  test_images_dir = os.path.join(dataset_path, "test_images", "test_images")
  
  
  results = model.predict(
      source=test_images_dir,
      augment=True,
      conf=0.01,
      iou=0.45,
      imgsz=1280,
      stream=True
  )
  
  rows = []
  for r in results:
      image_name = Path(r.path).name
      boxes_list = []
  
  
      for box, conf, cls_id in zip(r.boxes.xywhn, r.boxes.conf, r.boxes.cls):
          if int(cls_id) != 1: continue
          
  
          xc, yc, w, h = map(float, box)
          boxes_list.append([xc, yc, w, h, float(conf)])
          
      rows.append({"image_name": image_name, "boxes": json.dumps(boxes_list)})
  
  sub_df = pd.DataFrame(rows)
  sample_sub_path = os.path.join(dataset_path, "sample_sub.csv")
  sample_sub = pd.read_csv(sample_sub_path)
  final_sub = sample_sub[['id', 'image_name']].merge(sub_df, on='image_name', how='left')
  final_sub['boxes'] = final_sub['boxes'].fillna('[]')
  final_sub.to_csv(os.path.join(PROJECT_DIR, "submission.csv"), index=False)
  
  print(f"Сабмишен сохранен в: {os.path.join(PROJECT_DIR, 'submission.csv')}")

if __name__ == '__main__':
  main()
