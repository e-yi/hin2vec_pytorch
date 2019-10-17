# HIN2vec_pytorch
A pytorch implementation of [HIN2vec](https://github.com/csiesheep/hin2vec)


>*HIN2Vec* learns distributed representations of nodes in heterogeneous information networks (HINs) by capturing the distiguishing metapath relationships between nodes. 
Please refer the paper [here](https://dl.acm.org/citation.cfm?doid=3132847.3132953).

### Requirements
- python3.6+ 
- pytorch
- numpy
- pandas
- networkx

### Usage
create your own edge.csv referring to `demo_data.csv`

**The input graph is assumed to be undirected by default.**

create a main.py in the project folder, copy the following code and modify it.

```python
    import torch
    import pandas as pd
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    from walker import load_a_HIN_from_pandas
    from model import NSTrainSet, HIN2vec, train

    # set method parameters
    window = 4
    walk = 10
    walk_length = 300
    embed_size = 100
    neg = 5
    sigmoid_reg = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')

    # set dataset [PLEASE USE YOUR OWN DATASET TO REPLACE THIS]
    demo_edge = pd.read_csv('./demo_data.csv', index_col=0)

    edges = [demo_edge]

    print('finish loading edges')

    # init HIN
    hin = load_a_HIN_from_pandas(edges)
    hin.window = window

    dataset = NSTrainSet(hin.sample(walk_length, walk), hin.path_size, neg=neg)

    hin2vec = HIN2vec(hin.node_size, hin.path_size, embed_size, sigmoid_reg)

    # load model
    # hin2vec.load_state_dict(torch.load('hin2vec.pt'))

    # set training parameters
    n_epoch = 10
    batch_size = 20
    log_interval = 200

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(hin2vec.parameters())  # 原作者使用的是SGD？ 这里使用Adam
    loss_function = nn.BCELoss()

    for epoch in range(n_epoch):
        train(log_interval, hin2vec, device, data_loader, optimizer, loss_function, epoch)

    torch.save(hin2vec, 'hin2vec.pt')

    # set output parameters [the output file is a bit different from the original code.]
    node_vec_fname = 'node_vec.txt'
    # path_vec_fname = 'meta_path_vec.txt'
    path_vec_fname = None
    
    print(f'saving node embedding vectors to {node_vec_fname}...')
    node_embeds = pd.DataFrame(hin2vec.start_embeds.weight.data.numpy())
    node_embeds.rename(hin.id2node).to_csv(node_vec_fname, sep=' ')

    if path_vec_fname:
        print(f'saving meta path embedding vectors to {path_vec_fname}...')
        path_embeds = pd.DataFrame(hin2vec.path_embeds.weight.data.numpy())
        path_embeds.rename(hin.id2path).to_csv(path_vec_fname, sep=' ')
       
    # save model
    # torch.save(hin2vec.state_dict(), 'hin2vec.pt')
```

Some more details could be found in [the original repo](https://github.com/csiesheep/hin2vec)
