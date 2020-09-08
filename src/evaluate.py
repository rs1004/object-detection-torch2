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


def create_correct_table(iou: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """create a table of correctness and score

    Args:
        iou (torch.Tensor): (N, P, G)
        iou_thresh (float, optional): Iou threshold to be considered correct. Defaults to 0.5.

    Returns:
        torch.Tensor: (X, 2)
    """
    tensor = torch.max(iou, dim=1)
    result = []
    used = set()
    for i in range(len(tensor)):
        if tensor.values[i] == 0:
            break
        correct = False
        if tensor.values[i] > iou_thresh and tensor.indices[i] not in used:
            used.add(tensor.indices[i])
            correct = True
        result.append([correct, tensor.values[i]])
    return torch.Tensor(result)


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

    result_each_class = {c: {'result': torch.empty((0, 2)), 'count': torch.tensor(0)} for c in range(dataset.num_classes)}
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

                for c in result_each_class:
                    # outputs を score 順にソートする
                    sorted_outputs = torch.stack([outputs[n, torch.sort(outputs[:, :, c + 5]).indices[n], :] for n in range(args.batch_size)])

                    ious = calc_iou(sorted_outputs, gts)

                    # 対象外のものを 0 にする mask をかける
                    pr_mask = (torch.sum(sorted_outputs[:, :, 4:], dim=2, keepdims=True) != 0)
                    gt_mask = (torch.max(gts[:, :, 4:], dim=2).indices == c + 5).unsqueeze(1)

                    ious = ious * pr_mask * gt_mask

                    res = torch.cat([create_correct_table(iou) for iou in ious])
                    if len(res) > 0:
                        result_each_class[c]['result'] = torch.cat([result_each_class[c]['result'], res])
                    result_each_class[c]['count'] += torch.sum(gt_mask)

        # クラスごとの AP を計算
        result = {}
        for c in result_each_class:
            result = result_each_class[c]['result']
            count = result_each_class[c]['count']
            ap = calc_average_precision(result=result, count=count)
            result[c] = ap

        # レポート作成
        d = date.today().isoformat()

        runtime = check_output(['nvidia-smi']).decode()

        config_table = ['|item|value|', '|-|-|']
        for k, v in args.__dict__.items():
            config_table.append(f'|{k}|{v}|')

        score_table = ['|label|average precision|', '|-|-|']
        for class_id, ap in result.items():
            score_table.append(f'|{labelmap.id2name(class_id)}|{_float2str(ap.item())}|')

        m_ap = torch.stack(list(result.values())).mean()
        score_table.append(f'|**mean**|**{_float2str(m_ap.item())}**|')

        report = OUTPUT_FORMAT.format(
            date=d,
            runtime=runtime,
            config_table='\n'.join(config_table),
            score_table='\n'.join(score_table)
        )

        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.result_dir) / f'report_{d}.md', 'w') as f:
            f.write(report)

    print('Finished Evaluate')
