from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from enum import Enum
import xml.etree.ElementTree as ET
import json
import torch


class PascalVOCDataset(Dataset):
    def __init__(self, purpose, data_dirs, data_list_file_name, imsize, transform=None):
        self.transform = transform
        self.purpose = purpose
        if self.purpose not in Purpose.show_all():
            raise ValueError(f'purpose "{self.purpose}" is isvalid')
        self.imsize = imsize
        self.data_list = self._get_list(data_dirs, data_list_file_name)
        self.label_map = self._get_label_map()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if self.purpose == Purpose.CLASSIFICATION:
            class_name, coord, image_path = self.data_list[i]

            image = Image.open(image_path).crop(coord).resize((self.imsize, self.imsize))
            if self.transform:
                image = self.transform(image)
            gt = self.label_map[class_name]

        elif self.purpose == Purpose.DETECTION:
            image_path, gt_path = self.data_list[i]

            image = Image.open(image_path).resize((self.imsize, self.imsize))
            if self.transform:
                image = self.transform(image)
            gt = self._get_gt(gt_path)

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
                gt_path = Path(data_dir) / 'Annotations' / f'{i}.xml'
                if self.purpose == Purpose.CLASSIFICATION:
                    root = ET.parse(gt_path).getroot()
                    for obj in root.iter('object'):
                        class_name = obj.find('name').text
                        bbox = obj.find('bndbox')
                        coord = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)
                        data_list.append([class_name, coord, image_path])
                elif self.purpose == Purpose.DETECTION:
                    data_list.append([image_path, gt_path])

        return data_list

    def _get_label_map(self):
        label_map_path = Path(__file__).parent / 'labelmap.json'
        with open(label_map_path, 'r') as f:
            labels = json.load(f)['PascalVOC']
        label_map = {label: i for i, label in enumerate(labels)}
        return label_map

    def _get_gt(self, gt_path):
        class_num = len(self.label_map) + 1

        root = ET.parse(gt_path).getroot()
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


class Purpose(Enum):
    CLASSIFICATION = 'classification'
    DETECTION = 'detection'

    @classmethod
    def show_all(cls):
        return set(c.value for c in cls)
