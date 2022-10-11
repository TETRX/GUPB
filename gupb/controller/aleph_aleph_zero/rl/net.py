from torch import nn

class NeuralNetwork(nn.Module):

    def __init__(self, in_size, out_size):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = out_size
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.lin1 = nn.Linear(in_size, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.lin3 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(256, self.number_of_actions)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu1(out)
        out = self.lin2(out)
        out = self.relu2(out)
        out = self.lin3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out