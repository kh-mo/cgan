import os
import torch
import argparse
import numpy as np
from modeling import Generator
from torchvision.utils import save_image

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_z_dim', type=int, default=100)
    parser.add_argument("--label_size", type=int, default=10)
    parser.add_argument('--number', type=int, default=10)
    parser.add_argument("--track", default=True)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    G = Generator(args).to(device=args.device)

    # load pre-trained parameter
    G.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/generator")))

    # define noise distribution
    normal_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))

    # make save folder
    saved_folder = os.path.join(os.getcwd(), "evaluated_image")
    try:
        os.mkdir(saved_folder)
    except FileExistsError as e:
        pass

    # generate & save image
    if args.number == 10:
        z_candi = []
        for i in range(10):
            z = normal_distribution.sample(sample_shape=torch.Size([1, args.latent_z_dim])).to(device=args.device)
            for j in range(10):
                z_candi.append(z)
        z = torch.cat(z_candi, dim=0)
        label = torch.LongTensor([np.array([num for _ in range(10) for num in range(10)])]).squeeze(0)
        save_image(G(z, label), saved_folder + '/conditional_generated_100_images.png', nrow=10)
        print("image generated done.")
    else:
        z = normal_distribution.sample(sample_shape=torch.Size([10, args.latent_z_dim])).to(device=args.device)
        label = torch.LongTensor([np.array([n for n in range(10)])]).squeeze(0)
        save_image(G(z, label), saved_folder + '/conditional_generated_' + str(args.number) + ' images.png', nrow=10)
        print("image generated done.")

    print(G.parameters)
