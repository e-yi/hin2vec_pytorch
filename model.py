import torch
import torch.nn as nn
import torch.nn.functional as F


def toTensor(x, dtype):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    return x


class HIN2vec(nn.Module):

    def __init__(self, node_size, path_size, embed_dim, r=True):
        super().__init__()

        # self.args = args

        self.__initialize_model(node_size, path_size, embed_dim, r)

    def __initialize_model(self, node_size, path_size, embed_dim, r):
        self.start_embeds = nn.Embedding(node_size, embed_dim)
        self.end_embeds = self.start_embeds if r else nn.Embedding(node_size, embed_dim)

        self.path_embeds = nn.Embedding(path_size, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, start_node, end_node, path):
        start_node = toTensor(start_node, torch.long)
        end_node = toTensor(end_node, torch.long)
        path = toTensor(path, torch.long)

        s = self.start_embeds(start_node)
        e = self.end_embeds(end_node)
        p = self.path_embeds(path)
        p = torch.sigmoid(p)

        agg = torch.mul(s, e)
        agg = torch.mul(agg, p)
        # agg = F.sigmoid(agg)
        output = self.classifier(agg)

        return output


def train(log_interval, model, device, dataset, optimizer, loss_function, epoch):
    model.train()
    for idx, (data, target) in enumerate(dataset): # todo 引入batch
        # data, target =data.to(device), target.to(device) # disable gpu for now todo
        optimizer.zero_grad()
        output = model(*data)
        loss = loss_function(output, toTensor(target, torch.float).view((1,)))
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print(f'\rTrain Epoch: {epoch} '
                  f'[{idx * len(data)}/{len(dataset)} ({100. * idx / len(dataset):.3f}%)]\t'
                  f'Loss: {loss.item():.3f}\t'
                  f'data = {data}\t target = {target}',
                  end='')
    print()


if __name__ == '__main__':
    import pandas as pd
    from torch.utils.data import DataLoader
    import torch.optim as optim

    from utils import *
    from walker import format_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')

    edgeDiDr = pd.read_csv('G:/_projects/Lab/edge/edgeDiDr.csv', index_col=0)
    edgeDiSim = pd.read_csv('G:/_projects/Lab/edge/edgeDiSim.csv', index_col=0)
    edgeDrSim = pd.read_csv('G:/_projects/Lab/edge/edgeDrSim.csv', index_col=0)

    edgeDiSim_filtered = thFilter(edgeDiSim, th=0.7)
    edgeDrSim_filtered = thFilter(edgeDrSim, th=0.8)

    print('finish loading edges')

    hin = load_a_HIN_from_pandas([edgeDiDr, edgeDiSim_filtered, edgeDrSim_filtered])
    hin.window = 4

    data = list(format_data(hin.sample(300), hin.path_size, neg=3))
    np.random.shuffle(data)

    hin2vec = HIN2vec(hin.node_size, hin.path_size, 100)

    optimizer = optim.Adam(hin2vec.parameters())

    loss_function = nn.BCELoss()

    n_epoch = 5
    log_interval = 200

    for epoch in range(n_epoch):
        train(log_interval, hin2vec, device, data, optimizer, loss_function, epoch)

    torch.save(hin2vec, 'hin2vec.pt')
