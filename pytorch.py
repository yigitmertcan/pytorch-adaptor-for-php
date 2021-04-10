import torch
import argparse
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument("--type","-t")
veri = parser.parse_args()


def hello():
	x = torch.rand(5, 3)
	print(x)

def demo2():
	dtype = torch.float
	device = torch.device("cpu")
	x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
	y = torch.sin(x)
	a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
	b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
	c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
	d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

	learning_rate = 1e-6
	for t in range(2000):
	    y_pred = a + b * x + c * x ** 2 + d * x ** 3
	    loss = (y_pred - y).pow(2).sum()
	    if t % 100 == 99:
	        print(t, loss.item())

	    loss.backward()
	    with torch.no_grad():
	        a -= learning_rate * a.grad
	        b -= learning_rate * b.grad
	        c -= learning_rate * c.grad
	        d -= learning_rate * d.grad
	        a.grad = None
	        b.grad = None
	        c.grad = None
	        d.grad = None

	print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

def neural():
	class Net(nn.Module):

	    def __init__(self):
	        super(Net, self).__init__()
	        # 1 input image channel, 6 output channels, 3x3 square convolution
	        # kernel
	        self.conv1 = nn.Conv2d(1, 6, 3)
	        self.conv2 = nn.Conv2d(6, 16, 3)
	        # an affine operation: y = Wx + b
	        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
	        self.fc2 = nn.Linear(120, 84)
	        self.fc3 = nn.Linear(84, 10)

	    def forward(self, x):
	        # Max pooling over a (2, 2) window
	        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
	        # If the size is a square you can only specify a single number
	        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
	        x = x.view(-1, self.num_flat_features(x))
	        x = F.relu(self.fc1(x))
	        x = F.relu(self.fc2(x))
	        x = self.fc3(x)
	        return x

	    def num_flat_features(self, x):
	        size = x.size()[1:]  # all dimensions except the batch dimension
	        num_features = 1
	        for s in size:
	            num_features *= s
	        return num_features


	net = Net()
	print(net)


if veri.type == 'hello':
	hello()
elif veri.type == 'demo':
	demo2()
elif veri.type == 'neural':
	neural()
