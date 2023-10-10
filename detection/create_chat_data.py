import argparse
from pycocotools.coco import COCO
import json
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Create cat dataset')
    parser.add_argument('--data-root', default='cat_dataset', help='dataset root')
    parser.add_argument(
        '--img-dir', default='images', help='image folder path')
    parser.add_argument(
        '--ann-file',
        default='annotations/trainval.json',
        help='ann file path')
    parser.add_argument(
        '--out',
        '-o',
        type=str,
        default='cat_chat.jsonl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 读取数据
    coco = COCO(osp.join(args.data_root, args.ann_file))
    img_ids = coco.getImgIds()
    results = []

    for img_id in img_ids:
        raw_img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        raw_ann_info = coco.loadAnns(ann_ids)

        img_path = osp.join(args.data_root, args.img_dir, raw_img_info['file_name'])

        gt_bboxes = []
        for i, ann in enumerate(raw_ann_info):
            if ann.get('ignore', False):
                continue
            if ann.get('ignore', False):
                continue

            if ann.get('iscrowd', False):
                continue

            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, raw_img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, raw_img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            # 归一化
            bbox = [bbox[0] / raw_img_info['width'], bbox[1] / raw_img_info['height'],
                    bbox[2] / raw_img_info['width'], bbox[3] / raw_img_info['height']]

            # 乘以 1000
            bbox = [int(bbox[0] * 1000), int(bbox[1] * 1000), int(bbox[2] * 1000), int(bbox[3] * 1000)]
            gt_bboxes.append(bbox)

        if len(gt_bboxes) > 0:
            results.append({'img': img_path, 'bboxes': gt_bboxes})

    # print(results)

    # 转换为 jsonl 格式保存
    out_datas = []
    text = 'Please locate all the cat'
    prompt = 'Picture 1: <img>{}\n<ref>{}</ref><box>'

    for i, result in enumerate(results):
        out_dict = {'id': f"identity_{i}"}

        input_value = prompt.format(result['img'], text)

        gt_bbox_0 = result['bboxes'][0]

        target_value = f'({gt_bbox_0[0]},{gt_bbox_0[1]}),({gt_bbox_0[2]},{gt_bbox_0[3]})</box>'
        for bbox in result['bboxes'][1:]:
            target_value += f'<box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>'

        out_dict['conversations'] = [{'from': 'user', 'value': input_value}, {'from': 'assistant', 'value': target_value}]

        out_datas.append(out_dict)

    json.dump(out_datas, open(args.out, 'w'), indent=4, ensure_ascii=False)
