import argparse

import torch
from torch import nn, optim

from my_data import VOCAB, MyDataset, color_print
from my_models import MyModel0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-e", "--max_epoch", type=int, default=1000)
    parser.add_argument("-v", "--val-per", type=int, default=100)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = MyModel0(len(VOCAB), 16, 128).to(args.device)
    dataset = MyDataset("data/data_dict.pth", args.device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1, 1.2, 0.8], device=args.device))
    optimizer = optim.Adam(model.parameters())

    for _ in range(args.max_epoch // args.val_per):
        train(model, dataset, criterion, optimizer, max_epoch=args.val_per)
        validate(model, dataset)


def validate(model, dataset):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_val_data(batch_size=1)

        pred = model(text)

        for i, key in enumerate(keys):
            print_text, _ = dataset.val_dict[key]
            print_text_class = pred[:, i][: len(print_text)].cpu().numpy()
            color_print(print_text, print_text_class)

        # print(pred[:, 0])


def train(model, dataset, criterion, optimizer, max_epoch):
    model.train()

    for epoch in range(1, max_epoch + 1):
        optimizer.zero_grad()

        text, truth = dataset.get_train_data(batch_size=8)
        pred = model(text)

        loss = criterion(pred.view(-1, 4), truth.view(-1))
        loss.backward()

        optimizer.step()

        print(loss.item())


if __name__ == "__main__":
    main()
