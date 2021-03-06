"""
    
"""

import os
import argparse
import tqdm
import torch
import torchvision.transforms as transforms
from utils.energyDataset import EnergyDataset
from models.autoencoder import AutoEncoder_512, AutoEncoder_32
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(args: argparse.ArgumentParser) -> None:
    """
        training a model
    """
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # trainsform
    train_transform = transforms.Compose([
        # transforms.Resize((args.image_size, args.image_size,)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (0.5, 0.5, 0.5))
    ])

    # dataset and dataloader
    train_dataset = EnergyDataset(args.images_path, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # model
    if args.model_name == 'autoencoder_512':
        model = AutoEncoder_512(image_size=args.image_size, batch_size=args.batch_size, trainable=True)
    elif args.model_name == 'autoencoder_32':
        model = AutoEncoder_32(image_size=args.image_size, batch_size=args.batch_size, trainable=True)
    model.to(device)

    if args.weights != '':
        try:
            state_dict = torch.load(args.weights, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(e)

    # loss function
    losser = torch.nn.MSELoss()

    # optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.937, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Defined tensorboard summary.
    writer = SummaryWriter(log_dir=os.path.join(args.outputs, "runs", args.model_name))

    # Write graph.
    fake_img = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
    writer.add_graph(model, fake_img)

    for epoch in range(args.epochs):
        print(f"{epoch}/{args.epochs}")
        losses_train_mean = 0

        model.train()   # setting training mode.
        optimizer.zero_grad()   # clear grap.
        for _, (imgs, lab) in enumerate(tqdm.tqdm(train_dataloader, ncols=100)):
            # Load datas.
            imgs = imgs.to(device)

            # forward.
            outputs = model(imgs)

            # backward.
            losses = losser(outputs, imgs)
            losses.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad()

            # statictic.
            losses_train_mean += losses.item()

        # Update learning rate.
        scheduler.step()

        # Tensorboard summary.
        # losses curves.
        writer.add_scalars('Losses', {
            f'{args.model_name}_train' : losses_train_mean / len(train_dataloader)
        }, global_step=epoch)

        if args.checkpoint_save != '' and epoch % 10 == 0:
            os.makedirs(os.path.join(args.checkpoint_save, args.model_name), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_save, args.model_name, f"{args.model_name}_{epoch}_ck.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="what you want to do")
    parser.add_argument('--images-path', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/datas/images", help="images directory")
    parser.add_argument('--model-name', type=str, default='autoencoder_32', help="model name")
    parser.add_argument('--weights', type=str, default='', help="pre-training model")
    parser.add_argument('--batch-size', type=int, default=2, help="Traing batch size")
    parser.add_argument('--epochs', type=int, default=201, help="Epochs for training")
    parser.add_argument('--image-size', type=int, default=32, help="model input images size")
    parser.add_argument('--trainable', type=bool, default=True, help="True for traing or False for evel")
    parser.add_argument('--outputs', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/outputs", help="output directory")
    parser.add_argument('--checkpoint-save', type=str, default="/home/linsi/Projects/Energy Analysis/Unsupervised-Classification/weights", help="checkpoint saved path or None")
    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    train(args)