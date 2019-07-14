import os
import codecs
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols).type(torch.FloatTensor)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

class mytrainset:
    raw_folder = "raw_data"

    # read image, label file
    training_set = (
        read_image_file(os.path.join("./", raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join("./", raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join("./", raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join("./", raw_folder, 't10k-labels-idx1-ubyte'))
    )

    def __init__(self, train=True):
        self.data = None
        self.labels = None

        if train == True:
            self.data, self.labels = self.training_set
        else:
            self.data, self.labels = self.test_set

        self.data = (self.data - torch.min(self.data)) / (torch.max(self.data) - torch.min(self.data))

    # __getitem__, __len__은 python의 규약(protocol) 일종
    # duck typing : if it walks like a duck and it quacks like a duck, then it must be a duck.
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.layer1 = nn.Linear(self.args.latent_z_dim+self.args.label_size, 256)
        self.batch1 = nn.BatchNorm1d(256, track_running_stats=args.track)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 512)
        self.batch2 = nn.BatchNorm1d(512, track_running_stats=args.track)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 784)
        self.batch3 = nn.BatchNorm1d(784, track_running_stats=args.track)
        self.relu3 = nn.ReLU()

    def forward(self, x, label):
        onehot = torch.zeros(x.shape[0], self.args.label_size).scatter_(1, label.unsqueeze(1), 1).to(self.args.device)
        x = torch.cat((x, onehot), dim=1)
        x = self.layer1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.layer1 = nn.Linear(784+self.args.label_size, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, label):
        onehot = torch.zeros(x.shape[0], self.args.label_size).scatter_(1, label.unsqueeze(1), 1).to(self.args.device)
        x = x.view(-1, 784)
        x = torch.cat((x, onehot), dim=1)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.sigmoid1(x)
        return x

def save_images(image, epoch):
    saved_folder = os.path.join(os.getcwd(), "saved_image")
    try:
        os.mkdir(saved_folder)
    except FileExistsError as e:
        pass
    save_image(image, saved_folder+'/'+str(epoch+1)+'_epoch_image.png', nrow=10)

if __name__ == "__main__":
    ## hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_z_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--label_size", type=int, default=10)
    parser.add_argument("--track", default=True)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## load data
    md = mytrainset(train=True)
    train_loader = torch.utils.data.DataLoader(md, batch_size=args.batch_size, shuffle=False)

    ## model
    normal_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))
    G = Generator(args).to(device=args.device)
    D = Discriminator(args).to(device=args.device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)
    loss_function = torch.nn.BCELoss().to(device=args.device)

    ## training
    for epoch in range(args.epochs):
        g_loss_list = []
        d_loss_list = []
        for sample, label in train_loader:
            z = normal_distribution.sample(sample_shape=torch.Size([sample.size(0), args.latent_z_dim])).to(device=args.device)
            generated_image = G(z, label)

            g_loss = loss_function(D(generated_image, label), torch.ones_like(D(generated_image, label)))
            g_loss_list.append(g_loss.item())
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            d_loss_fake = loss_function(D(generated_image.detach(), label), torch.zeros_like(D(generated_image, label)))
            d_loss_real = loss_function(D(sample.to(args.device), label), torch.ones_like(D(generated_image, label)))
            d_loss = (d_loss_fake + d_loss_real) / 2
            d_loss_list.append(d_loss.item())
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        z_candi = []
        for i in range(10):
            z = normal_distribution.sample(sample_shape=torch.Size([1, args.latent_z_dim])).to(device=args.device)
            for j in range(10):
                z_candi.append(z)
        z = torch.cat(z_candi, dim=0)
        label = torch.LongTensor([np.array([num for _ in range(10) for num in range(10)])]).squeeze(0)
        print("epoch : {},\t g_loss : {},\t d_loss : {}".format(epoch, sum(g_loss_list)/len(g_loss_list), sum(d_loss_list)/len(d_loss_list)))
        save_images(G(z, label), epoch)

    # save model
    try:
        os.mkdir(os.path.join(os.getcwd(), "saved_model"))
    except FileExistsError as e:
        pass
    torch.save(G.state_dict(), os.path.join(os.getcwd(), "saved_model/generator"))
    torch.save(D.state_dict(), os.path.join(os.getcwd(), "saved_model/discriminator"))
