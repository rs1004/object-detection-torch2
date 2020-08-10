from utils import LabelMap
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from enum import Enum
import xml.etree.ElementTree as ET
import torch


class PascalVOCDataset(Dataset):
    def __init__(self, purpose: str, data_dirs: list, data_list_file_name: str, imsize: int, transform=None):
        """[summary]

        Args:
            purpose (str): 'classification' or 'detection'
            data_dirs (list): data dirs (string can also be set)
            data_list_file_name (str): ex) 'trainval.txt', 'test.txt'
            imsize (int): image size for resize
            transform (augmentation.Compose, optional): set of augmentation. Defaults to None.

        Raises:
            ValueError: occurs when 'purpose' is invalid
        """
        self.transform = transform
        self.purpose = purpose
        if self.purpose not in Purpose.get_all():
            raise ValueError(f'purpose "{self.purpose}" is isvalid')
        self.imsize = imsize
        self.data_list = self._get_list(data_dirs, data_list_file_name)
        self.labelmap = LabelMap('PascalVOC')
        self.num_classes = len(self.labelmap)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if self.purpose == Purpose.CLASSIFICATION.value:
            id, coord, image_path = self.data_list[i]

            image = Image.open(image_path).crop(coord).resize((self.imsize, self.imsize))
            gt = torch.eye(len(self.labelmap))[id]
            if self.transform:
                image, gt = self.transform(image, gt)

        elif self.purpose == Purpose.DETECTION.value:
            image_path, anno_path = self.data_list[i]

            image = Image.open(image_path).resize((self.imsize, self.imsize))
            gt = self._get_gt(anno_path)
            if self.transform:
                image, gt = self.transform(image, gt)

        return image, gt

    def _get_list(self, data_dirs: list, data_list_file_name: str) -> list:
        """get data list

        Args:
            data_dirs (list): data dirs (string can also be set)
            data_list_file_name (str): ex) 'trainval.txt', 'test.txt'

        Returns:
            list:
                * classification : [[class_id, bbox_coordinate, image_file_path], …]
                * detection      : [[image_file_path, annotation_file_path], …]
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        data_list = []
        for data_dir in data_dirs:
            data_list_path = Path(data_dir) / 'ImageSets' / 'Main' / data_list_file_name

            with open(data_list_path, 'r') as f:
                ids = f.read().split('\n')

            for i in ids[:-1]:
                image_path = Path(data_dir) / 'JPEGImages' / f'{i}.jpg'
                anno_path = Path(data_dir) / 'Annotations' / f'{i}.xml'
                if self.purpose == Purpose.CLASSIFICATION.value:
                    root = ET.parse(anno_path).getroot()
                    for obj in root.iter('object'):
                        id = self.labelmap.name2id(obj.find('name').text)
                        bbox = obj.find('bndbox')
                        coord = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)
                        data_list.append([id, coord, image_path])
                elif self.purpose == Purpose.DETECTION.value:
                    data_list.append([image_path, anno_path])

        return data_list

    def _get_gt(self, anno_path: Path) -> torch.Tensor:
        """get ground truth tensor

        Args:
            anno_path (Path): Annotation xml file path

        Returns:
            torch.Tensor: (G, 4 + C)
        """
        num_classes = self.num_classes + 1  # add void

        root = ET.parse(anno_path).getroot()
        gt = torch.empty(0, 4 + num_classes)
        for size in root.iter('size'):
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)
            coord = torch.Tensor([(xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height, (xmax - xmin) / width, (ymax - ymin) / height])
            id = self.labelmap.name2id(obj.find('name').text)
            score = torch.eye(num_classes)[id + 1]
            t = torch.cat([coord, score]).unsqueeze(0)
            gt = torch.cat([gt, t], dim=0)
        return gt


class Purpose(Enum):
    CLASSIFICATION = 'classification'
    DETECTION = 'detection'

    @classmethod
    def get_all(cls) -> set:
        """show all enum values

        Returns:
            set: all enum values
        """
        return set(c.value for c in cls)
