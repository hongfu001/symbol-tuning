import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset_train=pd.read_csv('/data/home/baixy/gaohongfu/ICL/SST-2/train.tsv', delimiter='\t')
dataset_val=pd.read_csv('/data/home/baixy/gaohongfu/ICL/SST-2/dev.tsv', delimiter='\t')
dataset_test=pd.read_csv('/data/home/baixy/gaohongfu/ICL/SST-2/test.tsv', delimiter='\t')

def symbol_create(content,label1,change1,label2,change2):
    label_change=[]
    for i in content:
        if i==label1:
            label_change.append("{m}".format(m=change1))
        if i==label2:
            label_change.append("{n}".format(n=change2))
    return label_change

train_symbol=symbol_create(dataset_train['label'],0,'1132 ', 1,'peter ')
val_symbol=symbol_create(dataset_val['label'],0,'1132 ', 1,'peter ')

def symbol_merge(sentence,symbol):
    content=[]
    input_str="Input:"
    label_str="Label:"
    for i in range(len(sentence)):
        content.append(input_str+sentence[i]+label_str+symbol[i])
    return content

def get_random_draw(lt, num):   
    lst= random.sample(range(0, lt), lt)
    paixu_list=[lst[i:i+num] for i in range(0, len(lst), num)]
    return paixu_list

def sampling(content,k):
    length=len(content)
    sampling_list_num=get_random_draw(length, k)
    sampling_list=[]
    for i in sampling_list_num:
        str1=''
        for j in i:
            str1="{m}{n}".format(m=str1,n=content[j])
        sampling_list.append(str1)
    return sampling_list

def sample_result(content,label,k):
    n=len(label)
    input_example_content=content[0:int(n*(k-1)/k)]
    input_example_label=label[0:int(n*(k-1)/k)]
    input_example=sampling(symbol_merge(input_example_content,input_example_label),k-1)

    lst=random.sample(range(int(n*(k-1)/k),n), n-int(n*(k-1)/k))
    label_total=[]
    input_total=[]
    for i in lst:
        label_total.append(label[i])
        prompt=content[i]
        example=input_example[(i-int(n*(k-1)/k))]
        input_total.append(example+prompt)
        
    return label_total,input_total
   
label_total,input_total=sample_result(dataset_train['sentence'],train_symbol,5)
data=pd.DataFrame({"content":input_total,"label":label_total})

print(data)