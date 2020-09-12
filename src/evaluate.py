from dataset import PascalVOCDataset
from model import SSD
from augmentation import Compose, ToTensor
from utils import collate_fn, calc_coordicate, calc_score, calc_iou, non_maximum_suppression, LabelMap
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
from datetime import date
from subprocess import check_output

OUTPUT_FORMAT = '''
# EVALUATION REPORT

## REPORTING DATE
{date}

## RUNTIME
```
{runtime}
```

## CONFIG
{config_table}

## SCORES
{score_table}
'''


def get_order(t: torch.Tensor, class_id: int) -> torch.Tensor:
    """get score order for class id

    Args:
        t (torch.Tensor): (X, C)
        class_id (int): class id

    Returns:
        torch.Tensor: (X',)
    """
    vals, indices = torch.sort(t[:, 5 + class_id], descending=True)
    return indices[vals > 0.]


def calc_average_precision(result: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """caluculate average precision

    Args:
        result (torch.Tensor): (X, 2)
        count (torch.Tensor): (1,)

    Returns:
        torch.Tensor: (1,)
    """
    correct = torch.sort(result, dim=0, descending=True).values[:, 0]

    TP = torch.cumsum(correct == 1., dim=0)
    FP = torch.cumsum(correct == 0., dim=0)

    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / count

    mod_precision = torch.cat([torch.Tensor([0.]), precision, torch.Tensor([0.])])
    mod_precision = torch.flip(torch.cummax(torch.flip(mod_precision, dims=[0]), dim=0).values, dims=[0])
    mod_recall = torch.cat([torch.Tensor([0.]), recall, torch.Tensor([1.])])

    return torch.sum(mod_precision[1:] * (mod_recall[1:] - mod_recall[:-1]))


def _float2str(val):
    return str(round(val, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
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

    labelmap = LabelMap('PascalVOC')

    i = 0
    result_correct = {}
    result_count = {c: 0 for c in range(dataset.num_classes)}
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for images, gts in pbar:
                # description
                pbar.set_description('[Evaluate]')

                # to GPU device
                images = images.to(device)
                gts = gts.to(device)

                # evaluate
                outputs = net(images)
                outputs[:, :, :4] = calc_coordicate(pr=outputs, df=defaults)
                outputs[:, :, 4:] = calc_score(pr=outputs)
                outputs = non_maximum_suppression(outputs=outputs)
                ious = calc_iou(outputs, gts)

                for output, gt, iou in zip(outputs, gts, ious):
                    result_correct[i] = dict()
                    for c in range(dataset.num_classes):
                        pr_order, gt_order = get_order(output, c), get_order(gt, c)
                        if len(pr_order) == len(gt_order) == 0:
                            continue
                        elif len(pr_order) == 0:
                            result_count[c] += len(gt_order)
                            continue
                        elif len(gt_order) == 0:
                            correct = torch.zeros(len(pr_order), 1).to(device)
                        else:
                            iou_one_class = iou[pr_order][:, gt_order]
                            valid = torch.eye(len(gt_order))[iou_one_class.max(dim=1).indices].to(device) * (iou_one_class > 0.5)
                            correct = ((valid.cumsum(dim=0) == valid) * valid).sum(dim=1, keepdims=True)
                        result_correct[i][c] = torch.cat([correct, output[pr_order][:, [5 + c]]], dim=1)
                        result_count[c] += len(gt_order)
                    i += 1

        # クラスごとの AP を計算
        result_dict = {}
        for c in range(dataset.num_classes):
            result = torch.cat([r[c] for _, r in result_correct.items() if c in r])
            count = result_count[c]
            ap = calc_average_precision(result=result, count=count)
            result_dict[c] = ap.to('cpu')

        # レポート作成
        d = date.today().isoformat()

        runtime = check_output(['nvidia-smi']).decode()

        config_table = ['|item|value|', '|-|-|']
        for k, v in args.__dict__.items():
            config_table.append(f'|{k}|{v}|')

        score_table = ['|label|average precision|', '|-|-|']
        for class_id, ap in result_dict.items():
            score_table.append(f'|{labelmap.id2name(class_id)}|{_float2str(ap.item())}|')

        m_ap = torch.stack(list(result_dict.values())).mean()
        score_table.append(f'|**mean**|**{_float2str(m_ap.item())}**|')

        report = OUTPUT_FORMAT.format(
            date=d,
            runtime=runtime,
            config_table='\n'.join(config_table),
            score_table='\n'.join(score_table)
        )

        with open(out_dir / f'report_{d}.md', 'w') as f:
            f.write(report)

    print('Finished Evaluate')
