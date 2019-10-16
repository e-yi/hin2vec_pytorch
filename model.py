import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

    def forward(self, start_node: torch.LongTensor, end_node: torch.LongTensor, path: torch.LongTensor):
        # assert start_node.dim() == 1  # shape = (batch_size,)

        s = self.start_embeds(start_node)  # (batch_size, embed_size)
        e = self.end_embeds(end_node)
        p = self.path_embeds(path)
        p = torch.sigmoid(p)

        agg = torch.mul(s, e)
        agg = torch.mul(agg, p)
        # agg = F.sigmoid(agg)
        output = self.classifier(agg)

        return output

    def output_embeddings(self, s_file_name, e_file_name, path_file_name):
        np.savetxt(s_file_name, self.start_embeds.weight.data.numpy())
        np.savetxt(e_file_name, self.end_embeds.weight.data.numpy())
        np.savetxt(path_file_name, self.path_embeds.weight.data.numpy())


def train(log_interval, model, device, train_loader: DataLoader, optimizer, loss_function, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data[:, 0], data[:, 1], data[:, 2])
        loss = loss_function(output.view(-1), target)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print(f'\rTrain Epoch: {epoch} '
                  f'[{idx * len(data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.3f}%)]\t'
                  f'Loss: {loss.item():.3f}\t\t',
                  # f'data = {data}\t target = {target}',
                  end='')
    print()


class NSTrainSet(Dataset):
    """
    完全随机的负采样 todo 改进一下？
    """

    def __init__(self, sample, path_size, neg=5):
        """
        :param path_size: 元路径总数
        :param neg: 负采样数目
        :param sample: HIN.sample()返回值，(start_node, end_node, path_id)
        """

        l = len(sample)

        x = np.tile(sample, (neg + 1, 1))
        y = np.zeros(l * (1 + neg))
        y[:l] = 1

        x[l:, 2] = np.random.randint(0, path_size - 1, (l * neg,))

        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)
        self.length = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import pandas as pd
    import torch.optim as optim

    from walker import load_a_HIN_from_pandas

    # set parameters
    window = 4
    walk_length = 300
    embed_size = 100
    neg = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')

    # set dataset [PLEASE USE YOUR OWN DATASET TO REPLACE THIS]
    demo_edge = pd.read_csv('./demo_data.csv', index_col=0)

    edges = [demo_edge]

    print('finish loading edges')

    # init HIN
    hin = load_a_HIN_from_pandas(edges)
    hin.window = window

    dataset = NSTrainSet(hin.sample(walk_length), hin.path_size, neg=neg)

    hin2vec = HIN2vec(hin.node_size, hin.path_size, embed_size)

    # set parameters
    n_epoch = 10
    batch_size = 20
    log_interval = 200

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(hin2vec.parameters())  # 原作者使用的是SGD？ 这里使用Adam
    loss_function = nn.BCELoss()

    for epoch in range(n_epoch):
        train(log_interval, hin2vec, device, data_loader, optimizer, loss_function, epoch)

    torch.save(hin2vec, 'hin2vec.pt')

    hin2vec.output_embeddings('start_node_embed.txt', 'end_node_embed.txt', 'path_embed.txt')
