from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# get dataset
train_dataset = datasets.ImageFolder(root="../data/train", transform=train_transforms)
test_dataset = datasets.ImageFolder(root="../data/test", transform=test_transforms)

# get dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)



# testing
def get_train_images():
    for images, labels in train_loader:
        plt.imshow(images[0].permute(1, 2, 0))
        plt.show()

def get_train_loader_shape():
    for images, labels in train_loader:
        # torch.Size([64, 3, 224, 224])
        print(images.shape)
        print(labels.shape)
        break

