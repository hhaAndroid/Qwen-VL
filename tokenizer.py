import os
os.environ['HF_MODULES_CACHE'] = './'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True, cache_dir='./qwen-7b-vl')

query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
    {'text': 'Generate the caption in English with grounding:'},
])
# Picture 1:<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>
# Generate the caption in English with grounding:
print(query)
# 这个过程实际上是调用： tokenize + convert_tokens_to_ids
inputs = tokenizer(query, return_tensors='pt')
# 输入图片名也是要编码的，并且会保证长度是 256，不够的会在后面追加 image_pad_tag
print(inputs)
