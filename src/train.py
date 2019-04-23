from my_data import MyDataset, VOCAB
from my_models import MyModel0

from torch import nn, optim

def main():
    model = MyModel0(len(VOCAB), 16, 128)
    dataset = MyDataset("data/data_dict.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(model, dataset, criterion, optimizer, max_epoch=10)


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
