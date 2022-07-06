# create data so that api is correct

import json
import numpy as np
import torch

mock_ft = np.linspace(0.4, 0.99, 101)
mock_sup = np.linspace(0.97, 0.66, 101)

mock_data = {}

for i in range(mock_ft.shape[0]):
    mock_data[i] = {}
    mock_data[i]['ft'] = mock_ft[i].item()
    mock_data[i]['sup'] = mock_sup[i].item()

mock_instance_data = {}

for prefix in ['ft', 'sup']:
    for j in range(3):
        mock_instance_data[f'{prefix}_{j}'] = {}

        for i in range(mock_ft.shape[0]):
            un_norm = np.random.uniform(0.0, 50.0, 10)
            un_norm /= np.sum(un_norm)
            mock_instance_data[f'{prefix}_{j}'][i] = torch.softmax(torch.Tensor(un_norm)/0.07, 0).tolist()

with open('data/mock/overall.json', 'w') as f:
    json.dump(mock_data, f)

with open('data/mock/instance.json', 'w') as f:
    json.dump(mock_instance_data, f)