from dataset import PascalVOCDataset
from model import SSD
from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import argparse


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
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--class_num', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--weights_path', type=str, default='./ssd_net.pth')
    parser.add_argument('--min_loss_path', type=str, default='./min_loss.txt')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = PascalVOCDataset(
        data_dirs=['/work/data/VOCdevkit/VOC2007', '/work/data/VOCdevkit/VOC2012'],
        data_list_file_name='trainval.txt',
        imsize=args.imsize,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        collate_fn=collate_fn)

    net = SSD(
        num_classes=args.class_num,
        weights_path=args.weights_path,
    )

    if Path(args.min_loss_path).exists():
        print('min_loss loaded.')
        with open(Path(args.min_loss_path), 'r') as f:
            min_loss = float(f.readlines()[0])
    else:
        min_loss = None

    optimizer = optim.SGD(net.train_params(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    defaults = net.default_bboxes.to(device)

    running_loss = 0.0
    for epoch in range(args.epochs):
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for i, (images, gts) in enumerate(pbar):
                # description
                pbar.set_description(f'[Epoch {epoch+1}/{args.epochs}] loss: {running_loss}')

                # to GPU device
                images = images.to(device)
                gts = gts.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = net.loss(pred_bboxes=outputs, default_bboxes=defaults, gt_bboxes=gts)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            if (min_loss is None) or (running_loss < min_loss):
                torch.save(net.state_dict(), args.weights_path_vgg16)
                min_loss = running_loss
                with open(Path(args.min_loss_path), 'w') as f:
                    f.write(str(min_loss))
            running_loss = 0.0

    print('Finished Training')
