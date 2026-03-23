from models.CustomNet import CustomNet
import torch
from torch import nn
from data.dataloader import get_dataloader

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo...
        # pred and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print information
        if batch_idx % 500 == 0:
          print(f'batch: {batch_idx}, loss: {loss}')

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'\nTrain Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

if __name__ == "__main__":
    train_loader, _ = get_dataloader()
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10

    for epoch in range(num_epochs):
        train(epoch, model, train_loader, criterion, optimizer)

    torch.save(model.state_dict(), './checkpoint/model.pth')