from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import json
import torch


class PascalVOCDataset(Dataset):
    def __init__(self, purpose, data_dirs, data_list_file_name, imsize, transform=None):
        self.transform = transform
        self.purpose = purpose
        self.imsize = imsize

        if self.purpose == 'classification':
            self.data_list = self._get_list_for_classification(data_dirs, data_list_file_name)
        else:
            raise ValueError(f'purpose "{self.purpose}" is isvalid')
        self.label_map = self._get_label_map()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if self.purpose == 'classification':
            class_name, coord, image_path = self.data_list[i]

            image = Image.open(image_path).crop(coord).resize((self.imsize, self.imsize))
            if self.transform:
                image = self.transform(image)
            label = torch.eye(len(self.label_map))[self.label_map[class_name]]

            return image, label

    def _get_label_map(self):
        label_map_path = Path(__file__).parent / 'labelmap.json'
        with open(label_map_path, 'r') as f:
            labels = json.load(f)['PascalVOC']
        label_map = {label: i for i, label in enumerate(labels)}
        return label_map

    def _get_list_for_classification(self, data_dirs, data_list_file_name):
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

                root = ET.parse(label_path).getroot()
                for obj in root.iter('object'):
                    class_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    coord = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)

                    data_list.append([class_name, coord, image_path])

        return data_list
