import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile
from play import start


data = pd.read_csv("./train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000]
def create_label(num):
    arr = np.zeros(10)
    arr[num] = 1
    return arr
Y_dev = torch.Tensor(np.array([create_label(data[y][0]) for y in range(1000)]))
X_dev = torch.Tensor(np.array([data[x][1:] for x in range(1000)]))

X_dev = X_dev / 255.

test_data = [(x, y) for x, y in zip(X_dev, Y_dev)]


data_dev = data[1000:m]
Y_train = torch.Tensor(np.array([create_label(data[y][0]) for y in range(m - 1000)]))
X_train = torch.Tensor(np.array([data[x][1:] for x in range(m - 1000)]))

X_train = X_train / 255.

train_data = [(x, y) for x, y in zip(X_train , Y_train)]


def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


class Network(nn.Module):
    def __init__(self) -> None: 
        super(Network, self).__init__()
        self.input_layer = nn.Linear(784, 128)
        self.hidden_layer = nn.Linear(128, 48)
        self.hidden_layer2 = nn.Linear(48, 16)
        self.output_layer = nn.Linear(16, 10)
        self.relu = nn.ReLU()
    def forward(self, X):

        X = self.relu(self.input_layer(X))
        X = self.relu(self.hidden_layer(X))
        X = self.relu(self.hidden_layer2(X))
        return self.output_layer(X)


if isfile("model.pt"):
    model = Network()
    model.load_state_dict(torch.load("model.pt"))
else:
    model = Network()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
def train(epoche):
    model.train()
    criterion = nn.CrossEntropyLoss() 
    loss = 0
    for batch_id, (data, target) in enumerate(train_data):
        print_progress_bar(batch_id, len(train_data), prefix=f'Epoche {epoche}:', suffix=f"Completed | Loss: {loss:.4f}", length=50)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

def create_image(data):
    image = data.reshape((28, 28)) / 255
    plt.style.use('grayscale')
    return image

def test():
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data = Variable(data)
            target = Variable(target)
            optimizer.zero_grad()
            out = model(data)

            prediction_class = torch.argmax(out)
            true_class = torch.argmax(target)
            if true_class.item() == prediction_class.item():
                correct += 1
            else:
                loss += 1

    print(f"Accuracy {(correct/(loss + correct))* 100}%")
    print(f"Loss: {loss}")
    print(f"Correct: {correct}")
def make_prediction(arr):
    arr = torch.Tensor(arr.T)
    data = Variable(arr)
    with torch.no_grad():
        optimizer.zero_grad()
        out = model(data)
        pred_class = torch.argmax(out)
        return pred_class.item()



for i in range(1, 20):
    train(i)
    test()

torch.save(model.state_dict(), "model.pt")

start(make_prediction)
