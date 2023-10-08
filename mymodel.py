import torch
import torchvision
from data_preparation import train_dataset, class_names
from PIL import Image
from Data_Prep import CustomDataLoader
from torch import nn
from tqdm.auto import tqdm

img, label = train_dataset[0]
# print(img.shape)
# print(label)
# print(class_names[label])

data = CustomDataLoader(train_root='S:/Projects/Data Science And ML/Accident Detection/Code/data/train',
                        test_root='S:/Projects/Data Science And ML/Accident Detection/Code/data/test',
                        batch_size=32)
#
# print(len(data.train_dataloader))
# print(len(data.test_dataloader))

batch, X = next(iter(data.train_dataloader))
# print(batch)
# print(X)

model = torchvision.models.resnet152(weights='DEFAULT')
# print(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


torch.manual_seed(42)

epochs = 20

for epoch in tqdm(range(epochs)):
    print(f"Epoch:{epoch}\n....")

    train_loss = 0
    for img, label in data.train_dataloader:
        model.train()
        pred = model(img)

        loss = loss_fn(pred, label)

        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data.train_dataloader)

    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for imgt, labelt in data.test_dataloader:
            test_pred = model(imgt)
            test_loss += loss_fn(test_pred, labelt)
            test_acc += accuracy_fn(y_true=labelt, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(data.test_dataloader)

        test_acc /= len(data.test_dataloader)

    print(f"\nTrain Loss:{train_loss:.4f} | Test Loss:{test_loss:.4f} | Test Accuracy:{test_acc:.4f}")


torch.save(model.state_dict(), 'S:/Projects/Data Science And ML/Accident Detection/Code/models/myModel.pth')
