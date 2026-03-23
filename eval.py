from models.CustomNet import CustomNet
import torch
from torch import nn
from data.dataloader import get_dataloader


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # pred and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy


if __name__ == "__main__":
    _, val_loader = get_dataloader()
    model = CustomNet().cuda()
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    criterion = nn.CrossEntropyLoss()

    validate(model, val_loader, criterion)