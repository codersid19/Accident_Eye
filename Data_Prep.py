import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class CustomDataLoader:
    def __init__(self, train_root, test_root, batch_size=32):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the train and test datasets
        self.train_dataset = datasets.ImageFolder(root=train_root, transform=self.transform)
        self.test_dataset = datasets.ImageFolder(root=test_root, transform=self.transform)

        # Get class names and indices
        self.class_names = self.train_dataset.classes
        self.class_idx = self.train_dataset.class_to_idx

        # Create data loaders
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)

# Example usage:
# data_loader = CustomDataLoader(train_root='path_to_train_data', test_root='path_to_test_data', batch_size=32)
# You can then access data_loader.train_dataloader and data_loader.test_dataloader in another .py file.
