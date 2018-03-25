import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class Net(nn.Module):
    """Construct network."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)


# params = list(net.parameters())
# print(len(params))
# print(params[0].size())
# print(params[0].requires_grad)

inpt = Variable(torch.randn(1, 1, 32, 32))
out = net(inpt)
# print(out)

target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()


net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

lr = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * lr)
