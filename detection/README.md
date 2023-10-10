# 检测项目说明

##  cat 数据集评估

必须要 pytorch 2.0

```shell
pip install tiktoken
pip install transformers_stream_generator
pip install accelerate
pip install mmengine
pip install pycocotools
```

数据集必须要下载或者软链接到 `detection/` 目录下，目前是以最简单的 cat 数据集为例

```shell
cd detection
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip -d cat_dataset
rm -rf cat_dataset.zip
```

在准备好数据集后，因为这个数据集在 pillow 读取时候会出现宽高反的情况，直接评估性能会很低，因此需要先对数据集进行简单处理

```shell
python process_data.py
```

然后就可以开始进行评估

```shell
# 单卡
python evaluate_detection.py --data-root cat_dataset
# 分布式
python -m torch.distributed.launch --nproc_per_node=8 evaluate_detection.py --data-root cat_dataset --launcher pytorch 
```

运行完会输出 mAP 同时会在 outputs 里面保存一个 `test_pred.json`，你可以使用 `browse_coco_json.py` 脚本可视化分析效果

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.766
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.931
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.844
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.800
```

性能比较好的原因是数据集比较简单，同时大部分图片里面都是只有一个物体，模型在没有训练过的情况下无法输出多个 bbox。


