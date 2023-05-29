import torch as T
import torch.nn as nn

class LeNet5(nn.Module):
    
    def __init__(self, activation: str = 'ReLU')->None:
        """Create LeNet-5 model with custom activation functions

        Args:
            activation (str, optional): Type of activation function to use. Defaults to 'ReLU'.
        """
        super(LeNet5, self).__init__()
        
        # Instantiate LeNet-5 Model with custom activation functions
        self.cnn_model: nn.Sequential = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=(5,5)),
            self.activation_factory(activation),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            self.activation_factory(activation),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5,5)),
            self.activation_factory(activation),
        )
        
        self.fc_model: nn.Sequential = nn.Sequential(
            nn.Linear(120, 84),
            self.activation_factory(activation),
            nn.Linear(84, 10),
            nn.Softmax(-1)
        )
        
    def forward(self, img: T.Tensor)->T.Tensor:
        """Generate Prediction based on image fed through model
        
        Input must be of size (1,32,32)
        """
        output = self.cnn_model(img)
        output = self.fc_model(output.view(1,-1))
        return output
    
    def activation_factory(self, activation: str)->nn.Module:
        """Return an instance of an activation function depending on what is requested

        Args:
            activation (str): Type of activation function

        Returns:
            nn.Module: instance of activation function
        """
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_ReLU':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
        