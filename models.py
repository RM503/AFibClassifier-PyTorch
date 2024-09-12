'''
This is a model file for the AlexNet architecture class. The AlexNet architecture consists of five convolutional blocks followed by three 
fully connected linear layers. In each sequential layer, the output shape of the tensor are annotated in comments.
'''

import torch.nn as nn 


class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Five convolutional layers

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),          # output shape : 96 x 55 x 55
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                           # output shape : 96 x 27 x 27
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),         # output shape : 256 x 27 x 27
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                           # output shape : 256 x 13 x 13
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),        # output shape : 384 x 13 x 13
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),        # output shape : 384 x 13 x 13
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),        # output shape : 256 x 13 x 13
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                           # output shape : 256 x 6 x 6
        )

        # Three fully connected layers; starting input shape is 256 x 6 x 6 = 9216

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1, 9216)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
    

