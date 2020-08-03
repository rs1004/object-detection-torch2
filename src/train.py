from dataset import PascalVOCDataset, Purpose
from model import VGG16, SSD
from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import argparse
import json


def collate_fn(batch):
    images = []
    gts = []
    for image, gt in batch:
        images.append(image)
        gts.append(gt)
    images = torch.stack(images, dim=0)
    gts = pad_sequence(gts, batch_first=True)
    return images, gts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--purpose', type=str, default='detection')
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--weights', type=str, default='weights.pth')
    parser.add_argument('--params', type=str, default='params.json')
    args = parser.parse_args()

    weights_path = Path(args.result_dir) / 'train' / args.purpose / args.weights
    params_path = Path(args.result_dir) / 'train' / args.purpose / args.params

    transform = transforms.Compose([
        transforms.ToTensor()])

    dataset = PascalVOCDataset(
        purpose=args.purpose,
        data_dirs=['/work/data/VOCdevkit/VOC2007', '/work/data/VOCdevkit/VOC2012'],
        data_list_file_name='trainval.txt',
        imsize=args.imsize,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.purpose == Purpose.CLASSIFICATION.value:
        net = VGG16(
            num_classes=dataset.num_classes,
            weights_path=weights_path
        )
        loss_args = {}
    elif args.purpose == Purpose.DETECTION.value:
        net = SSD(
            num_classes=dataset.num_classes + 1,  # add void
            weights_path=weights_path,
            weights_path_vgg16=Path(args.result_dir) / 'train' / 'classification' / args.weights
        )
        defaults = net.default_bboxes.to(device)
        loss_args = {'default_bboxes': defaults}
    net.to(device)

    if params_path.exists():
        print('Params loaded.')
        with open(params_path, 'r') as f:
            params = json.load(f)
        min_loss = params['min_loss']
        lr = params['lr']
    else:
        min_loss = None
        lr = args.lr

    optimizer = optim.Adam(net.train_params(), lr=lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)

    running_loss = 0.0
    for epoch in range(args.epochs):
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for i, (images, gts) in enumerate(pbar, start=1):
                # description
                pbar.set_description(f'[Epoch {epoch+1}/{args.epochs}] loss: {running_loss/i}')

                # to GPU device
                images = images.to(device)
                gts = gts.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss_args.update({'outputs': outputs, 'targets': gts})
                loss = net.loss(**loss_args)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            running_loss /= i
            if (min_loss is None) or (running_loss < min_loss):
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                # save weights
                torch.save(net.state_dict(), weights_path)
                # save params
                params = {'min_loss': running_loss, 'lr': scheduler.get_last_lr()[0]}
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=4)
            running_loss = 0.0

    print('Finished Training')
