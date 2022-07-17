import os, pickle, random
import numpy as np
import torch

random.seed(1)

data = {}

for p in sorted(os.listdir('data/real/logits'))[::-1]:
    with open(f'data/real/logits/{p}', 'rb') as f:
        alpha, dset, identifier = p.split('=')[-1].split('_')
        read = pickle.load(f)
        if dset not in data:
            # print(dset)
            data[dset] = {}
        if alpha not in data[dset]:
            # print(alpha)
            data[dset][alpha] = {}
        if identifier not in data[dset][alpha]:
            # print(identifier)
            data[dset][alpha][identifier] = {}
        # print(read.shape)
        data[dset][alpha][identifier] = read
        if identifier == 'logits':
            data[dset][alpha]['preds'] = np.argmax(read, axis=1)
            # print(dset, alpha, np.sum(data[dset][alpha]['preds'] == data[dset][alpha]['ys']) / data[dset][alpha]['preds'].shape[0])

alphas = ['0.00', '0.33', '0.66', '0.99']
# MNIST
# print(data)
# Mnist wrong at 0 and right at 0.33 onwards
mnist_index_to_correctness = {}
for alpha in alphas:
    correctness = data['MNIST'][alpha]['preds'] == data['MNIST'][alpha]['ys']
    for i in range(correctness.shape[0]):
        if i not in mnist_index_to_correctness:
            mnist_index_to_correctness[i] = []
        mnist_index_to_correctness[i].append(correctness[i].item())

# print(mnist_index_to_correctness)
good_mnist_samples = []
mnist_target_pattern = [False, True, True, True]
for i in mnist_index_to_correctness:
    if mnist_index_to_correctness[i] == mnist_target_pattern:
        good_mnist_samples.append(i)

# CIFAR
# CIFAR 1 right always
# CIFAR 2 right always
# CIFAR 3 right until 0.99
cifar_index_to_correctness = {}
for alpha in alphas:
    correctness = data['CIFAR10'][alpha]['preds'] == data['CIFAR10'][alpha]['ys']
    for i in range(correctness.shape[0]):
        if i not in cifar_index_to_correctness:
            cifar_index_to_correctness[i] = []
        cifar_index_to_correctness[i].append(correctness[i].item())

good_cifar_samples = []
cifar_target_pattern = [True, True, True, True]
for i in cifar_index_to_correctness:
    if cifar_index_to_correctness[i] == cifar_target_pattern:
        good_cifar_samples.append(i)

medium_cifar_samples = []
cifar_target_pattern = [True, True, True, False]
for i in cifar_index_to_correctness:
    if cifar_index_to_correctness[i] == cifar_target_pattern:
        medium_cifar_samples.append(i)

# print(medium_cifar_samples)

mnist1 = random.choice(good_mnist_samples)
mnist2 = random.choice(good_mnist_samples)
mnist3 = random.choice(good_mnist_samples)

cifar1 = random.choice(good_cifar_samples)
cifar2 = random.choice(good_cifar_samples)
cifar3 = random.choice(medium_cifar_samples)

print('mnist')
for i in (mnist1, mnist2, mnist3):
    # print(data['MNIST']['0.00']['ys'][i])
    print(i)

print('cifar')
for i in (cifar1, cifar2, cifar3):
    print(i)
    # print(data['CIFAR10']['0.00']['ys'][i])

instance_data = {}

for prefix in ['ft', 'sup']:
    for j in range(3):
        instance_data[f'{prefix}_{j}'] = {}

for i, a in  enumerate(np.linspace(0., 1., 101)):
    alpha_str = f'{a:.2f}'
    if alpha_str in data['MNIST']:
        instance_data['ft_0'][str(i)] = torch.softmax(torch.Tensor(data['MNIST'][alpha_str]['logits'][mnist1]), 0).tolist()
        instance_data['ft_1'][str(i)] = torch.softmax(torch.Tensor(data['MNIST'][alpha_str]['logits'][mnist2]), 0).tolist()
        instance_data['ft_2'][str(i)] = torch.softmax(torch.Tensor(data['MNIST'][alpha_str]['logits'][mnist3]), 0).tolist()

for i, a in  enumerate(np.linspace(0., 1., 101)):
    alpha_str = f'{a:.2f}'
    if alpha_str in data['CIFAR10']:
        instance_data['sup_0'][str(i)] = torch.softmax(torch.Tensor(data['CIFAR10'][alpha_str]['logits'][cifar1]), 0).tolist()
        instance_data['sup_1'][str(i)] = torch.softmax(torch.Tensor(data['CIFAR10'][alpha_str]['logits'][cifar2]), 0).tolist()
        instance_data['sup_2'][str(i)] = torch.softmax(torch.Tensor(data['CIFAR10'][alpha_str]['logits'][cifar3]), 0).tolist()



# print(instance_data)