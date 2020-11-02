import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputSize = 28
sequenceLength = 28
numLayers = 2

hiddenSize = 128
numClasses = 10
epochs = 2
batchSize = 100
learningRate = 0.001


trainDataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
testDataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, numClasses):
        super(LSTM, self).__init__()
        self.numLayers = numLayers
        self.hiddenSize = hiddenSize
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)

        self.fc = nn.Linear(hiddenSize, numClasses)

    def forward(self, x):
        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(device)

        c0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTM(inputSize, hiddenSize, numLayers, numClasses).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

nTotalSteps = len(trainLoader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(trainLoader):
        images = images.reshape(-1, sequenceLength, inputSize).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], step [{i+1}/{nTotalSteps}], loss: {loss.item():.4f}')

with torch.no_grad():
    nCorrect = 0
    nSamples = 0
    for images, labels in testLoader:
        images = images.reshape(-1, sequenceLength, inputSize).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        nSamples += labels.size(0)
        nCorrect += (predicted == labels).sum().item()

    accuracy = 100.0 * nCorrect/nSamples
    print(f'accuracy of the net work on 1000 test images: {accuracy} %')
