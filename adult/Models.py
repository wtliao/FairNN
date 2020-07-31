import torch
import torch.nn as nn
import torch.nn.functional as F


"""
autoencoder structure:
encoder:107-54-32-10
decoder:107-54-32-10
"""
class MyAutoencoder(nn.Module):
    def __init__(self):
        super(MyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(99, 54),
            nn.Tanh(),
            nn.Linear(54, 32),
            nn.Tanh(),
            nn.Linear(32, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 54),
            nn.Tanh(),
            nn.Linear(54, 99),
            nn.ReLU()
        )

        # for p in self.parameters():
        #     p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        clf_result = self.classifier(encode)
        return encode, decode, clf_result


"""
classifier model
"""
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.Classifier = nn.Linear(10, 1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.Classifier.weight.data)
        self.Classifier.bias.data.zero_()

    def forward(self, x):
        output_data = self.Classifier(x)
        return torch.sigmoid(output_data)


class Classifier_1(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.hidden_layer = nn.Linear(10, 20)
        self.predict_layer = nn.Linear(20, 1)

    def forward(self, x):
        hidden_result = self.hidden_layer(x)
        relu_result = F.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return torch.sigmoid(predict_result)
