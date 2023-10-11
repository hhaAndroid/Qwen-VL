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

如果是 chat 模型，则采用如下命令

```shell
# 单卡
python evaluate_detection.py --data-root cat_dataset --model Qwen/Qwen-VL-Chat --cache-dir ../qwen-7b-vl-chat
# 分布式
python -m torch.distributed.launch --nproc_per_node=8 evaluate_detection.py --data-root cat_dataset --model Qwen/Qwen-VL-Chat --cache-dir ../qwen-7b-vl-chat --launcher pytorch 
```

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.759
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.931
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.844
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.759
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.800
```

## 微调训练

```shell
pip install peft deepspeed
```

deepspeed 在第一次运行时候会编译 cuda op，因此需要提前设置好 CUDA_HOME ncvv 和 gcc 否则会报错

**(1) 生成 jsonl 对话格式数据**

```shell
python create_chat_data.py
```

**(2) 开启训练**

```shell
sh finetune_lora_ds.sh
```

官方代码直接跑会报错：

```text
    hidden_states[i][a + 1 : b] = images[idx]
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
```

需要对代码进行修改，但是因为用的是远程代码，为啥使得修改后代码生效同时方便 debug，可以把运行时候缓存的数据换个位置

在 `finetune.py` 代码最前面加上

```python
import os
os.environ['HF_MODULES_CACHE'] = './'
```        

然后就可以手动修改 `transformers_modules/Qwen/Qwen-VL-Chat/2562bb20c375615e99b422d425066af6939d165e/modeling_qwen.py` 内部代码。

将 655 行代码修改，错误原因是不允许这样直接 inplace。

```python
if images is not None:
    hs_list = []
    for idx, (i, a, b) in enumerate(img_pos):
        head = hidden_states[i][:a + 1, :]
        tail = hidden_states[i][b:, :]
        hs_list.append(torch.cat([head, images[idx], tail], dim=0))
    hidden_states = torch.stack(hs_list, dim=0)
```

模型训练完成后会在当前 detection 路径中生成一个 `output_qwen` 文件夹，内部是 lora 权重和对应配置

## 微调后推理

单张图片可视化直接运行 ../qwen-vl.py 和 ../qwen-vl-chat.py 即可，修改内部的 model_style 参数即可

评估脚本如下：

```shell
# 单卡
python evaluate_detection.py --data-root cat_dataset --model output_qwen --model-style lora --cache-dir ../qwen-7b-vl-chat
# 分布式
python -m torch.distributed.launch --nproc_per_node=8 evaluate_detection.py --data-root cat_dataset --model output_qwen --model-style lora --cache-dir ../qwen-7b-vl-chat --launcher pytorch 
```

## 微调非 chat 模型

现在已经支持了非 chat 格式和模型微调。需要注意：

chat 版本额外训练了 <|im_start|> 和 <|im_end|> 这两个 token，用户对话的，如果想微调非 chat 版本，但是你用了这两个 token，那么就需要设置 modules_to_save = ["wte", "lm_head"]，让 embedding 和 输出层可以训练，会增加很多显存。但是如果我非 chat 版本微调和推理中都不存在这两个，那么就也不需要

因此我们修改了 finetune 脚本。




