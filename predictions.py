import torch
import torchvision
from data_preparation import train_dataset, class_names
from PIL import Image
from Data_Prep import CustomDataLoader
from torch import nn
from tqdm.auto import tqdm

model = torchvision.models.resnet152(weights='DEFAULT')
checkpoint = torch.load('S:/Projects/Data Science And ML/Accident Detection/Code/models/myModel.pth')
model.load_state_dict(checkpoint)

data = CustomDataLoader(train_root='S:/Projects/Data Science And ML/Accident Detection/Code/data/train',
                        test_root='S:/Projects/Data Science And ML/Accident Detection/Code/data/test',
                        batch_size=32)

class_name = train_dataset.classes


def accuracy_fn(y_true, y_pred):
    # Assuming y_true is a 1D tensor (torch.Size([32])) and y_pred is a 2D tensor (torch.Size([32, 1000]))

    # Convert y_pred to class predictions by selecting the class with the highest probability
    predicted_labels = y_pred.argmax(dim=1)

    # Compare predicted_labels with y_true
    correct = torch.eq(y_true, predicted_labels).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


batch_acc = []

# Set the model to evaluation mode
model.eval()

with torch.no_grad():
    for inputs, labels in data.test_dataloader:
        outputs = model(inputs)
        acc = accuracy_fn(labels, outputs)
        batch_acc.append(acc)

test_accuracy = torch.tensor(batch_acc).mean().item()
print(f"Test accuracy of Model is {test_accuracy:.4f}")
