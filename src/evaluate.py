from dataset import PascalVOCDataset
from model import SSD
from augmentation import Compose, ToTensor
from utils import collate_fn
from pathlib import Path
from tqdm import tqdm
import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--weights', type=str, default='weights.pth')
    args = parser.parse_args()

    weights_path = Path(args.result_dir) / 'detection' / args.weights
    out_dir = Path(args.result_dir) / 'detection'
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = Compose([
        ToTensor()])

    dataset = PascalVOCDataset(
        purpose='detection',
        data_dirs='/work/data/VOCdevkit/VOC2007',
        data_list_file_name='test.txt',
        imsize=args.imsize,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SSD(
        num_classes=dataset.num_classes + 1,
        weights_path=weights_path,
        weights_path_vgg16=Path(args.result_dir) / 'classification' / args.weights
    )
    net.to(device)
    defaults = net.default_bboxes.to(device)

    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for images, _ in pbar:
                # description
                pbar.set_description('[Evaluate]')

                # to GPU device
                images = images.to(device)

                # evaluate
                outputs = net(images)

                # TBD

    print('Finished Evaluate')
