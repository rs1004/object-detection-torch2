from dataset import PascalVOCDataset, Purpose
from model import VGG16, SSD
from utils import collate_fn
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import augmentation as aug
import torch.optim as optim
import torch
import argparse
import json


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

    weights_path = Path(args.result_dir) / args.purpose / args.weights
    params_path = Path(args.result_dir) / args.purpose / args.params

    transform = aug.Compose([
        aug.RandomColorJitter(p=0.5),
        aug.RandomPerspective(p=0.5),
        aug.RandomFlip(p=0.5),
        aug.RandomScale(p=0.5),
        aug.RandomShift(p=0.5),
        aug.ToTensor(),
        aug.RandomErasing(p=0.5, max_iter=3)])

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
            weights_path_vgg16=Path(args.result_dir) / 'classification' / args.weights
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
        start_epoch = params['last_epoch']
    else:
        min_loss = None
        lr = args.lr
        start_epoch = 0

    optimizer = optim.Adam(net.train_params(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    writer = SummaryWriter(log_dir='./logs')

    running_loss = 0.0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1 + start_epoch, args.epochs + start_epoch + 1):
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for i, (images, gts) in enumerate(pbar, start=1):
                # description
                pbar.set_description(f'[Epoch {epoch}/{args.epochs + start_epoch}] loss: {running_loss / i}')

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

                running_loss += loss.item()

            running_loss /= i
            writer.add_scalar('loss', running_loss, epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            if (min_loss is None) or (running_loss < min_loss):
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                # save weights
                torch.save(net.state_dict(), weights_path)
                # save params
                params = {'min_loss': running_loss, 'lr': scheduler.get_last_lr()[0], 'last_epoch': epoch}
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=4)

            running_loss = 0.0
            scheduler.step()

    print('Finished Training')

    writer.close()
