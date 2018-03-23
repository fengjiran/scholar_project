from __future__ import print_function

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
# x.requires_grad = True
print(x)

y = x + 2
print(y.requires_grad)

z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)

a = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
