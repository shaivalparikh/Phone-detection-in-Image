import sys
import numpy as np
import cv2
import torch
import torch.nn as nn


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


def predict(image, PATH):
    """
    This function predicts the coordinates of the phone in the input image using the trained model

    :param image: Loaded and normalized image
    :param PATH: Path to the saved model
    """
    learningRate = 0.001
    weightDecay = 0.00005
    in_channel = 3
    feat_dim = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()

    network = Model(in_channel, feat_dim)  # Initializing the model
    network = network.to(device)

    checkpoint = torch.load(PATH)

    optimizer = torch.optim.Adam(network.parameters(), lr=learningRate, weight_decay=weightDecay)
    # Initializing the optimizer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=7, factor=0.2, threshold=0.01,
                                                           verbose=True)  # Initializing the scheduler

    network.load_state_dict(checkpoint['model_state_dict'])  # Reload the saved and trained model weights and parameters
    optimizer.load_state_dict(
        checkpoint['optimizer_state_dict'])  # Reload the saved and trained optimizer parameters
    scheduler.load_state_dict(
        checkpoint['scheduler_state_dict'])  # Reload the saved and trained scheduler parameters

    network = network.to(device)

    output = network(image)  # Passing the image into the trained model

    if device == torch.device('cuda'):
        output = output.to('cpu')

    op_list = output.tolist()

    print(round(op_list[0][0], 4), round(op_list[0][1], 4))


if __name__ == "__main__":
    image_path = sys.argv[1]
    PATH = "./models/best_model.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = cv2.imread(image_path)
    resize = cv2.resize(img, (64, 64))
    img = np.asarray(resize)
    image = np.interp(img, (img.min(), img.max()), (0, 1))
    image = torch.FloatTensor(image)
    image = torch.permute(image, (2, 0, 1))
    image = image.to(device)
    image = image.unsqueeze(0)

    predict(image, PATH)
