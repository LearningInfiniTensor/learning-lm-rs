from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# excute from project directory.
model_directory = "models/story"

model = AutoModelForCausalLM.from_pretrained(model_directory)

print(model.config)

tokenizer = AutoTokenizer.from_pretrained(model_directory)
text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")
outputs_dict = {}

for name, param in model.named_parameters():
    print(f"Name: {name}, Size: {param.size()}, Type: {param.dtype}")
    if name == 'model.layers.0.post_attention_layernorm.weight':
        print(param.detach().numpy()[:])


def hook_fn(layer_name):
    def hook(module, input, output):
        if layer_name not in outputs_dict:
            outputs_dict[layer_name] = []
        outputs_dict[layer_name].append({
            "input": input,
            "output": output
        })
    return hook
    

# 注册钩子
for name, layer in model.named_modules():
    layer_name = f"transformer_layer_{name}"
    layer.register_forward_hook(hook_fn(layer_name))

# 执行推理
with torch.no_grad():
    model(**inputs)


for layer_name, data_list in outputs_dict.items():
    print(f"Layer: {layer_name}")

    for data in data_list:
        # 打印输入形状
        if isinstance(data['input'], tuple):
            for t in data['input']:
                if isinstance(t, torch.Tensor):
                    print(f"Input shape: {t.shape}")
        elif isinstance(data['input'], torch.Tensor):
            print(f"Input shape: {data['input'].shape}")
        else:
            print(f"Input type: {type(data['input'])}")

        # 打印输出形状
        if isinstance(data['output'], tuple):
            for t in data['output']:
                if isinstance(t, torch.Tensor):
                    print(f"Output shape: {t.shape}")
        elif isinstance(data['output'], torch.Tensor):
            print(f"Output shape: {data['output'].shape}")
        else:
            print(f"Output type: {type(data['output'])}")

        print(f"Input: {data['input']}")
        print(f"Output: {data['output']}")
        print()