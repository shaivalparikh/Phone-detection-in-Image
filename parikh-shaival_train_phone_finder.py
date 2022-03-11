import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class dataset(Dataset):
    """
    This is a class to format the input files and label pairs and convert them to tensors and then into batches of
    the defined size.
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def collate_fn(batch):
        x = np.asarray([x[0] for x in batch])
        y = np.array([x[1] for x in batch])
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        x = torch.permute(x, (0, 3, 1, 2))
        return x, y


class Model(nn.Module):
    """
    This is the class where the model used for training has been defined It contains 3 Convolutional Layers and 2
    Fully connected linear layers with activation layers between them Batch normalization, max pooling and dropout
    layers have also been added after convolutional layers to improve the performance of the model
    """

    def __init__(self, input_shape, feat_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_shape, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, (2, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, (2, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, (2, 2)),
            nn.Dropout(0.3),

            nn.Flatten(),

            nn.Linear(4096, 256),
            nn.Sigmoid(),

            nn.Linear(256, 128),
            nn.Sigmoid()

        )
        self.linear = nn.Linear(128, feat_dim)

    def forward(self, x):
        embedding = self.layers(x)
        embedding = self.linear(embedding)
        return embedding


def dataloader(image_path):
    """
    This function takes in the paths to the images and the labels file and processes the files present at these paths
    and converts them to usable formats for the models The images are first read and then resized to a square size
    numpy array. Since the coordinate values are between 0 and 1, the image array values are interpolated between 0 and
    1 to normalize the distances so that they can be used by the model for learning.
    The data is then split into train and test data and converted into dataloader format using the dataset class
    The function returns the train_dataloader, test_dataloader and the length of the test dataset

    :param image_path: Path to images folder
    :return: train_dataloader, test_dataloader and test_data_len
    """
    label_data = []
    labels = image_path + "/labels.txt"
    
    with open(labels) as file:
        for line in file:
            data = [l.strip() for l in line.split(' ')]
            label_data.append(data)
    image = []
    coord = []

    for data in label_data:
        img = cv2.imread(image_path + "/" + data[0])
        resize = cv2.resize(img, (64, 64))
        image.append(resize)
        coord.append([float(data[1]), float(data[2])])

    image = np.asarray(image)
    coord = np.asarray(coord)

    image = np.interp(image, (image.min(), image.max()), (0, 1))

    (X_train, x_test, Y_train, y_test) = train_test_split(image, coord, test_size=0.2, random_state=42)

    batch_size = 8
    num_workers = 4

    train_data = dataset(X_train, Y_train)
    train_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn,
                      pin_memory=True)
    train_dataloader = DataLoader(train_data, **train_args)

    test_data = dataset(x_test, y_test)
    test_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn,
                     pin_memory=True)
    test_dataloader = DataLoader(test_data, **test_args)
    test_data_len = len(test_data)
    return train_dataloader, test_dataloader, test_data_len


def train_model(train_dataloader, test_dataloader,test_data_len):
    """
    This function trains and validates the model defined above and saves the model weights and parameters in .pt file

    :param train_dataloader: Dataloader containting the training image data and corresponding labels split into batches
    :param test_dataloader: Dataloader containting the testing image data and corresponding labels split into batches
    :param test_data_len: Length of the test dataset
    """
    numEpochs = 200
    feat_dim = 2

    learningRate = 0.001
    weightDecay = 0.00005
    in_channel = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()

    net = Model(in_channel, feat_dim)
    net = net.to(device)

    criterion = nn.MSELoss()  # Loss function

    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay=weightDecay)
    # Optimizer for the model parameters

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=7, factor=0.2, threshold=0.01,
                                                           verbose=True)  # Scheduler for learning rate optimization

    best_acc = 0
    for epoch in range(numEpochs):

        net.train()
        avg_loss = 0.0

        for batch, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            output = net(x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        net.eval()
        num_correct = 0
        for batch, (x, y) in enumerate(test_dataloader):

            x, y = x.to(device), y.to(device)

            outputs = net(x)

            err = abs(outputs - y)
            for i in range(len(y)):
                if torch.dist(outputs[i],y[i],p=2) <= 0.05:
                    num_correct += 1

        val_acc = num_correct / test_data_len
        print("Epoch: {}\tValidation Acc: {}".format(epoch+1, val_acc * 100))

        if val_acc >= best_acc:
            best_acc = val_acc
            PATH = "./models/best_model.pt"  # Path to save the model
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'validation_accuracy': best_acc
                        }, PATH)

    print('Training Complete')


if __name__ == "__main__":
    image_path = str(sys.argv[1])
    
    train_dataloader, test_dataloader, test_data_len = dataloader(image_path)

    train_model(train_dataloader, test_dataloader, test_data_len)