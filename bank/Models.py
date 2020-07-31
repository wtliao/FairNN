import torch
import torch.nn as nn
import torch.nn.functional as F

"""
autoencoder structure:
encoder:50-25-13-10
decoder:10-13-25-50
"""
class MyAutoencoder(nn.Module):
    def __init__(self):
        super(MyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 13),
            nn.Tanh(),
            nn.Linear(13, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 13),
            nn.Tanh(),
            nn.Linear(13, 25),
            nn.Tanh(),
            nn.Linear(25, 50),
            nn.ReLU()
        )
        self.fine_tuning = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        clf_result = self.fine_tuning(encode)

        return encode, decode, clf_result


"""
classifier model
"""

class Classifier_1(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.hidden_layer = nn.Linear(5, 10)
        self.predict_layer = nn.Linear(10, 1)

    def forward(self, x):
        hidden_result = self.hidden_layer(x)
        relu_result = F.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return torch.sigmoid(predict_result)


class Classifier_2(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.hidden_layer = nn.Linear(50, 100)
        self.predict_layer = nn.Linear(100, 1)

    def forward(self, x):
        hidden_result = self.hidden_layer(x)
        relu_result = F.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return torch.sigmoid(predict_result)