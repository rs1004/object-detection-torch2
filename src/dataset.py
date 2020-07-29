from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import json
import torch


class PascalVOCDataset(Dataset):
    def __init__(self, data_dirs, data_list_file_name, imsize, transform=None):
        self.transform = transform
        self.imsize = imsize
        self.data_list = self._get_list(data_dirs, data_list_file_name)
        self.label_map = self._get_label_map()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        image_path, label_path = self.data_list[i]

        image = Image.open(image_path).resize((self.imsize, self.imsize))
        if self.transform:
            image = self.transform(image)
        gt = self._get_gt(label_path)

        return image, gt

    def _get_list(self, data_dirs, data_list_file_name):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        data_list = []
        for data_dir in data_dirs:
            data_list_path = Path(data_dir) / 'ImageSets' / 'Main' / data_list_file_name

            with open(data_list_path, 'r') as f:
                ids = f.read().split('\n')

            for i in ids[:-1]:
                image_path = Path(data_dir) / 'JPEGImages' / f'{i}.jpg'
                label_path = Path(data_dir) / 'Annotations' / f'{i}.xml'
                data_list.append([image_path, label_path])

        return data_list

    def _get_label_map(self):
        label_map_path = Path(__file__).parent / 'labelmap.json'
        with open(label_map_path, 'r') as f:
            labels = json.load(f)['PascalVOC']
        label_map = {label: i for i, label in enumerate(labels)}
        return label_map

    def _get_gt(self, label_path):
        class_num = len(self.label_map) + 1

        root = ET.parse(label_path).getroot()
        gt = torch.empty(0, 4 + class_num)
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)
            offset = torch.Tensor([(xmin + xmax)/2, (ymin + ymax)/2, xmax - xmin, ymax - ymin]) / self.imsize
            label_id = self.label_map[obj.find('name').text]
            score = torch.eye(class_num)[label_id + 1]
            t = torch.cat([offset, score]).unsqueeze(0)
            gt = torch.cat([gt, t], dim=0)
        return gt
