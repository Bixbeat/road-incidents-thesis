import torch.nn as nn

def generate_classifiers(num_classes, criterion=nn.CrossEntropyLoss):
    classifiers = {}
    for i in range(num_classes):
        classifiers['{}'.format(i)] = criterion()
    return classifiers

class ResNet2FCL(nn.Module):
    #constructor
    def __init__(self,resnet,fc_layers):
        super().__init__()
        #defining layers in convnet
        self.r=resnet
        self.fc1 = fc_layers[0]
        self.fc2 = fc_layers[1]

    def forward(self, x):
        x = self.r.conv1(x)
        x = self.r.bn1(x)
        x = self.r.relu(x)
        x = self.r.maxpool(x)

        x = self.r.layer1(x)
        x = self.r.layer2(x)
        x = self.r.layer3(x)
        x = self.r.layer4(x)

        x = self.r.avgpool(x)
        x = x.view(x.size(0), -1)
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)

        return [fc1, fc2]






