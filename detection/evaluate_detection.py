import re
import argparse
import json
import os
import warnings
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmengine.dataset import DefaultSampler, worker_init_fn
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.utils import ProgressBar

from transformers import AutoModelForCausalLM, AutoTokenizer

debug = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen-VL')  # Qwen/Qwen-VL or Qwen/Qwen-VL-Chat
    parser.add_argument('--cache-dir', default='../qwen-7b-vl')  # ../qwen-7b-vl or ../qwen-7b-vl-chat
    parser.add_argument('--model-style', default='normal')  # normal lora
    parser.add_argument('--data-root', type=str, default='cat_dataset')
    parser.add_argument('--data-prefix', type=str, default='images/')
    parser.add_argument(
        '--ann-file', type=str, default='annotations/test.json')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class SimpleDataset(Dataset):

    def __init__(self, data_root, data_prefix, coco, tokenizer, prompt, text='Please locate all the cat'):
        self.coco = coco
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.img_ids = coco.getImgIds()
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.text = text

    def __getitem__(self, idx):
        raw_img_info = self.coco.loadImgs([self.img_ids[idx]])[0]
        file_name = raw_img_info['file_name']
        image_path = os.path.join(self.data_root, self.data_prefix, file_name)  # TODO
        w, h = raw_img_info['width'], raw_img_info['height']

        return {
            'text': self.prompt.format(image_path, self.text),
            'img_id': self.img_ids[idx],
            'hw': (h, w),
        }

    def __len__(self):
        return len(self.img_ids)


def collate_fn(batches, tokenizer):
    texts = [_['text'] for _ in batches]
    img_ids = [_['img_id'] for _ in batches]
    hws = [_['hw'] for _ in batches]

    input_ids = tokenizer(texts, return_tensors='pt', padding='longest')

    return input_ids.input_ids, input_ids.attention_mask, img_ids, hws


if __name__ == '__main__':
    args = parse_args()

    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True

    if _distributed and not is_distributed():
        init_dist(args.launcher)

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if not debug:
        if args.model_style == 'normal':
            model = AutoModelForCausalLM.from_pretrained(
                args.model, device_map='cuda', trust_remote_code=True, cache_dir=args.cache_dir).eval()
        elif args.model_style == 'lora':
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model,
                device_map="cuda",
                cache_dir=args.cache_dir,
                trust_remote_code=True
            ).eval()
        else:
            raise NotImplementedError()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir=args.cache_dir)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    prompt = 'Picture 1:<img>{}</img>\n<ref>{}</ref>'

    coco = COCO(os.path.join(args.data_root, args.ann_file))
    coco_dataset = SimpleDataset(args.data_root, args.data_prefix, coco, tokenizer, prompt)

    name2id = {}
    for categories in coco.dataset['categories']:
        name2id[categories['name']] = categories['id']

    if get_rank() == 0:
        print('data len: ', len(coco_dataset), 'num_word_size: ',
              get_dist_info()[1])

    sampler = DefaultSampler(coco_dataset, False)
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_worker,
        rank=get_rank(),
        seed=0,
        disable_subprocess_warning=True)
    data_loader = DataLoader(
        dataset=coco_dataset,
        sampler=sampler,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        worker_init_fn=init_fn,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        persistent_workers=False,
        drop_last=False)

    if get_rank() == 0:
        progress_bar = ProgressBar(len(data_loader))

    part_json_data = []

    PATTERN = re.compile(r'\((.*?)\),\((.*?)\)')

    for _, (input_ids, attention_mask, img_ids, hws) in tqdm(enumerate(data_loader)):
        if not debug:
            pred = model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=28,
                min_new_tokens=10,
                length_penalty=1,
                num_return_sequences=1,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
            )
        else:
            pred = [] * len(img_ids)

        for p, id, hw in zip(pred, img_ids, hws):
            new_json_data = dict(annotation=[])
            image_id = id
            raw_img_info = coco.loadImgs([image_id])[0]
            raw_img_info['img_id'] = image_id
            new_json_data['image'] = raw_img_info

            if not debug:
                answer = tokenizer.decode(p[input_ids.size(1):].cpu(), skip_special_tokens=True)
                print(answer)
                predict_bbox = re.findall(PATTERN, answer)
                print(predict_bbox)
            else:
                predict_bbox = [["(1,2)", "(3,4)"]]

            with_bbox = True
            try:
                if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][1]:
                    with_bbox = False
                else:
                    x1, y1 = [
                        float(tmp) for tmp in predict_bbox[0][0].split(',')
                    ]
                    x2, y2 = [
                        float(tmp) for tmp in predict_bbox[0][1].split(',')
                    ]
                    predict_bbox = [x1, y1, x2, y2]
                    print(predict_bbox)
            except:
                with_bbox = False

            if with_bbox:
                predict_bbox = np.array(predict_bbox, dtype=float).reshape(-1, 4) / 999
                predict_bbox[:, 0::2] *= hw[1]
                predict_bbox[:, 1::2] *= hw[0]

                predict_bbox = predict_bbox.tolist()

                for i in range(len(predict_bbox)):
                    bbox = predict_bbox[i]
                    coco_bbox = [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ]
                    annotation = dict(
                        image_id=image_id,
                        bbox=coco_bbox,
                        score=1.0,  # TODO
                        iscrowd=0,
                        category_id=name2id['cat'],  # always cat
                        area=coco_bbox[2] * coco_bbox[3])
                    annotation['segmentation'] = []
                    new_json_data['annotation'].append(annotation)
                part_json_data.append(new_json_data)
            else:
                part_json_data.append(new_json_data)
                continue

        if get_rank() == 0:
            progress_bar.update()

    all_json_results = collect_results(part_json_data, len(coco_dataset), 'cpu')

    if get_rank() == 0:
        new_json_data = {
            'info': coco.dataset.get('info', []),
            'licenses': coco.dataset.get('licenses', []),
            'categories': coco.dataset['categories'],
            'images':
                [json_results['image'] for json_results in all_json_results]
        }

        annotations = []
        annotation_id = 1
        for annotation in all_json_results:
            annotation = annotation['annotation']
            for ann in annotation:
                ann['id'] = annotation_id
                annotation_id += 1
                annotations.append(ann)

        if len(annotations) > 0:
            new_json_data['annotations'] = annotations

        output_json_name = args.ann_file[:-5] + '_pred.json'
        output_name = os.path.join(args.out_dir, output_json_name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        with open(output_name, 'w') as f:
            json.dump(new_json_data, f)

        if len(coco.dataset['annotations']) > 0:
            cocoDt = COCO(output_name)
            metrics = ['bbox']
            for metric in metrics:
                coco_eval = COCOeval(coco, cocoDt, iouType=metric)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
        else:
            warnings.warn("No gt label, can't evaluate")
