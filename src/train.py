from dataset import PascalVOCDataset
from model import VGG16
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_weights_path', type=str, default='./vgg16_net.pth')
    parser.add_argument('--min_loss_path', type=str, default='./vgg16_min_loss.txt')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])

    dataset = PascalVOCDataset(
        purpose='classification',
        data_dirs=['/work/data/VOCdevkit/VOC2007', '/work/data/VOCdevkit/VOC2012'],
        data_list_file_name='trainval.txt',
        imsize=args.imsize,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10)

    net = VGG16(
        class_num=dataset.class_num
    )

    if Path(args.model_weights_path).exists():
        print('weights loaded.')
        net.load_state_dict(torch.load(Path(args.model_weights_path)))

    if Path(args.min_loss_path).exists():
        print('min_loss loaded.')
        with open(Path(args.min_loss_path), 'r') as f:
            min_loss = float(f.readlines()[0])
    else:
        min_loss = None

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    running_loss = 0.0
    for epoch in range(args.epochs):
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for i, (images, labels) in enumerate(pbar):
                # description
                pbar.set_description(f'[Epoch {epoch+1}/{args.epochs}] loss: {running_loss}')

                # to GPU device
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = net.loss(output=outputs, target=labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            if (min_loss is None) or (running_loss < min_loss):
                torch.save(net.state_dict(), args.model_weights_path)
                min_loss = running_loss
                with open(Path(args.min_loss_path), 'w') as f:
                    f.write(str(min_loss))
            running_loss = 0.0

    print('Finished Training')
