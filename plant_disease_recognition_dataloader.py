from class_dataset import PlantImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
main_dir = "/Users/aravdhoot/Plant_Disease_Classification/data/Plant Disease Recognition"

train_dataset = PlantImageDataset(csv_file="plants_disease__recognition_train.csv", root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/Plant Disease Recognition/Train/Train", main_dir=main_dir, transform=transforms)
valid_dataset = PlantImageDataset(csv_file="plants_disease__recognition_valid.csv", root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/Plant Disease Recognition/Validation/Validation", main_dir=main_dir, transform=transforms)
test_dataset = PlantImageDataset(csv_file="plants_disease__recognition_test.csv", root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/Plant Disease Recognition/Test/Test", main_dir=main_dir, transform=transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)