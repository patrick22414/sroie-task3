import argparse

import torch
from torch import nn, optim

from my_data import VOCAB, MyDataset, color_print
from my_models import MyModel0
from my_utils import pred_to_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-e", "--max_epoch", type=int, default=1000)
    parser.add_argument("-v", "--val-at", type=int, default=100)
    parser.add_argument("-i", "--hidden-size", type=int, default=256)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)

    dataset = MyDataset("data/data_dict.pth", args.device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1, 1.2, 0.8], device=args.device))
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1000)

    for i in range(args.max_epoch // args.val_at):
        train(
            model,
            dataset,
            criterion,
            optimizer,
            (i * args.val_at + 1, (i + 1) * args.val_at + 1),
            args.batch_size,
        )
        validate(model, dataset)

    validate(model, dataset, batch_size=10)


def validate(model, dataset, batch_size=1):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_val_data(batch_size=batch_size)

        oupt = model(text)
        prob = torch.nn.functional.softmax(oupt, dim=2)
        pred = torch.argmax(prob, dim=2)

        for i, key in enumerate(keys):
            real_text, _ = dataset.val_dict[key]
            result = pred_to_dict(real_text, pred[:, i].cpu().numpy(), prob[:, i].cpu().numpy())
            for k, v in result.items():
                print(f"{k:>8}: {v}")
            # print_text_class = pred[:, i][: len(print_text)].cpu().numpy()
            # color_print(print_text, print_text_class)


def train(model, dataset, criterion, optimizer, epoch_range, batch_size):
    model.train()

    for epoch in range(*epoch_range):
        optimizer.zero_grad()

        text, truth = dataset.get_train_data(batch_size=batch_size)
        pred = model(text)

        loss = criterion(pred.view(-1, 4), truth.view(-1))
        loss.backward()

        optimizer.step()

        print(f"#{epoch:04d} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
