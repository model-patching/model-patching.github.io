# create data so that api is correct

import json
import numpy as np
import torch

mapping = {
    'CIFAR10': 'sup',
    'MNIST': 'ft'
}

with open('data/real/overall.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = {}

for json_str in json_list:
    result = json.loads(json_str)
    alpha_100 = int(round(result['alpha'] * 100))
    if alpha_100 not in data:
        data[alpha_100] = {}
    data[alpha_100][mapping[result["exp_name"]]] = result[f'{result["exp_name"]}:top1']

# mock_instance_data = {}

# for prefix in ['ft', 'sup']:
#     for j in range(3):
#         mock_instance_data[f'{prefix}_{j}'] = {}

#         for i in range(mock_ft.shape[0]):
#             un_norm = np.random.uniform(0.0, 50.0, 10)
#             un_norm /= np.sum(un_norm)
#             mock_instance_data[f'{prefix}_{j}'][i] = torch.softmax(torch.Tensor(un_norm)/0.07, 0).tolist()

with open('data/real/overall.json', 'w') as f:
    json.dump(data, f)

# with open('data/mock/instance.json', 'w') as f:
#     json.dump(mock_instance_data, f)