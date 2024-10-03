from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np


max_v_l = 1000000000
seed = 0
random.seed(seed)
np.random.seed(seed)

org_datalist = load_jsonl('data/highlight_train_release.jsonl')
clip_len = 2

mult_times = 5

# org_datalist = load_jsonl('data/tacos/train.jsonl')
# clip_len = 2

# org_datalist = load_jsonl('data/charades/charades_sta_train_tvr_format.jsonl')
# clip_len = 1


new_datalist = []
duration = 150

for data in org_datalist:

    new_data = deepcopy(data)
    
    duration = data['duration']
    new_data['duration'] = 0
    
    new_data['org_clip_ids_order'] = []
                                 
    for i in range(mult_times):
        new_data['org_clip_ids_order'].append([0, duration])
        new_data['duration'] += duration
    
    new_datalist.append(new_data)


print(f"Oracle Crop : {len(org_datalist)} -> {len(new_datalist)}")
save_jsonl(new_datalist, f'data/qv_{mult_times}.jsonl')
# save_jsonl(new_datalist, f'cha_new_new_aug.jsonl')
# save_jsonl(new_datalist, f'new_merge_tacos.jsonl')
