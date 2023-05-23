import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class givemodel():
    def __init__(self , train_path ,test_path, learning_rate, num_epochs):
        self.train_path = train_path
        self.test_path = test_path
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.transforms = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010],)
                    ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(
            torchvision.datasets.ImageFolder(train_path, transform=self.transforms),
            batch_size=64, shuffle=True
        )
        test_loader = DataLoader(
            torchvision.datasets.ImageFolder(test_path, transform=self.transforms),
            batch_size=32, shuffle=True)

        self.model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.loss_function = nn.CrossEntropyLoss()
        self.num_epochs = 20

        train_count = len(glob.glob(self.train_path + '/**/*.png'))
        test_count = len(glob.glob(self.test_path + '/**/*.png'))

        print(train_count, test_count)

        best_accuracy = 0.0

        for epoch in range(num_epochs):

            # Evaluation and training on training dataset
            self.model.train()
            train_accuracy = 0.0
            train_loss = 0.0

            for i, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_accuracy += int(torch.sum(prediction == labels.data))

            train_accuracy = train_accuracy / train_count
            train_loss = train_loss / train_count

            # Evaluation on testing dataset
            self.model.eval()

            test_accuracy = 0.0
            for i, (images, labels) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())

                outputs = self.model(images)
                _, prediction = torch.max(outputs.data, 1)
                test_accuracy += int(torch.sum(prediction == labels.data))

            test_accuracy = test_accuracy / test_count

            print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
                train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

            # Save the best model
            if test_accuracy > best_accuracy:
                torch.save(self.model.state_dict(), 'best_checkpoint1.model')
                best_accuracy = test_accuracy

    def save_onnx_model(self, name_of_model):

        dummy_input = torch.randn(1, 3, 224, 224, device="cuda")

        input_names = ["data"]
        output_names = ["resnetv24_dense0_fwd"]

        torch.onnx.export(self.model, dummy_input, name_of_model, verbose=False, input_names=input_names,
                          output_names=output_names)





if __name__ == '__main__':
    train_path=r"C:\Users\rohin\Desktop\torch\dataset\weld_train"
    test_path=r"C:\Users\rohin\Desktop\torch\dataset\weld_test"
    obj = givemodel(train_path,test_path,0.001,10)
    obj.save_onnx_model("restest.onnx")


