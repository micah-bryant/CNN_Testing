import torch as T
import torch.nn as nn

class LeNet5(nn.Module):
    
    def __init__(self, activation: str = 'ReLU'):
        super(LeNet5, self).__init__()
        self.model: nn.Sequential = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=(5,5)),
            self.activation_factory(activation),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            self.activation_factory(activation),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5,5)),
            self.activation_factory(activation),
            nn.Linear(120, 84),
            self.activation_factory(activation),
            nn.Linear(84, 10),
            nn.Softmax(10)
        )
        
    def forward(self, img):
        output = self.model(img)
        return output
        
    def activation_factory(self, activation: str)->nn.Module:
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_ReLU':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
        