from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score  
import torch,os
import pandas as pd
import sys
import time
import json
import torch, gc

gc.collect()
torch.cuda.empty_cache()

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/home/baixy/aimodel/chatglm2-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("/data/home/baixy/aimodel/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("/data/home/baixy/aimodel/chatglm2-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(path, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()


def predict(text):
    response,history=model.chat(tokenizer,"{text}->".format(text=text),history=[])
    return response

def result(text,label1,change_label1,label2,change_label2,label_true):
    predict_label=[]
    for i in text:
        predict_temp=(predict(i))
        if predict_temp==change_label1:
            predict_label.append(label1)
        else:
            predict_label.append(label2)
    
    accuracy=accuracy_score(label_true,predict_label)
    return accuracy
    
    
        