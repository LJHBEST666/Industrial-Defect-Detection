import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,input):
        print(f'input:{input.shape}')
        
        c1 = F.relu(self.conv1(input))
        print(f'c1:{c1.shape}')

        s2 = F.max_pool2d(c1,(2,2))
        print(f's2:{s2.shape}')

        c3 = F.relu(self.conv2(s2))
        print(f'c3:{c3.shape}')

        s4 = F.max_pool2d(c3,2)
        print(f's4:{s4.shape}')

        s4 = torch.flatten(s4,1)
        print(f's4:{s4.shape}')

        f5 = F.relu(self.fc1(s4))
        print(f'f5:{f5.shape}')

        f6 = F.relu(self.fc2(f5))
        print(f'f6:{f6.shape}')

        output = self.fc3(f6)
        return output
net = Net()

input = torch.randn(1,1,32,32)
out = net(input)
print(f'output:{out.shape}')