import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    # required
    # Dataset 객체가 생성될 때 최초 1회 실행. 이미지와 주석 파일이 포함된 디렉토리와 transform 초기화
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # required
    # Dataset 샘플의 개수 반환
    def __len__(self):
        return len(self.image_labels)

    # required
    # 주어진 idx에 해당하는 샘플을 Dataset에서 불러와서 반환.
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
