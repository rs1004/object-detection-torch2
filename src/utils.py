import json
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


def collate_fn(batch):
    images = []
    gts = []
    for image, gt in batch:
        images.append(image)
        gts.append(gt)
    images = torch.stack(images, dim=0)
    gts = pad_sequence(gts, batch_first=True)
    return images, gts


class LabelMap:
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.labels = self._get_labels()

    def _get_labels(self):
        label_map_path = Path(__file__).parent / 'labelmap.json'
        with open(label_map_path, 'r') as f:
            labels = json.load(f)[self.ds_name]
        return labels

    def __len__(self):
        return len(self.labels)

    def name2id(self, name):
        return self.labels.index(name)

    def id2name(self, id):
        return self.labels[id]
