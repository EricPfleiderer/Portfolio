import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.loaders import get_dataset_by_name
from src.classifier import get_classifier_by_dataset_name
import logging


class TorchTrainable:

    """
    Wrapper for torch models, their optimizer and their training procedure.
    """

    def __init__(self, params):

        logging.info('Initializing a trainable.')

        self.params = params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        logging.info(f'Model device:{device}')

        # Torch data loaders
        self.train_loader, self.test_loader = get_dataset_by_name(params['dataset_name'], params['batch_size'])

        # Torch model
        self.model = get_classifier_by_dataset_name(params['dataset_name']).to(self.device)

        # Torch optimizer
        self.optimizer = params['optimizer']['type'](self.model.parameters(), **params['optimizer']['opt_params'])

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

    def reset_history(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

    def train(self):

        logging.info(f'Training MNIST classifier for ' + str(self.params['num_epochs']) + ' epochs')

        self.reset_history()

        for epoch in range(self.params['num_epochs']):

            # Training
            train_loss = 0
            train_hits = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                predictions = output.data.max(1, keepdim=True)[1]
                train_hits += predictions.eq(target.data.view_as(predictions)).sum()
                train_loss = F.nll_loss(output, target)
                train_loss.backward()
                self.optimizer.step()
            # train_loss /= len(self.train_loader)  # implicit in .backward call?
            train_accuracy = train_hits / self.train_loader.dataset.data.shape[0]

            # Validation
            test_loss = 0
            test_hits = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    predictions = output.data.max(1, keepdim=True)[1]
                    test_hits += predictions.eq(target.data.view_as(predictions)).sum()

            test_loss /= len(self.test_loader.dataset)
            test_accuracy = test_hits / self.test_loader.dataset.data.shape[0]

            self.history['train_loss'].append(train_loss.item())
            self.history['train_accuracy'].append(train_accuracy.data.cpu().detach().numpy())
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_accuracy.data.cpu().detach().numpy())

            logging.info(f'Epoch: {epoch}/' + str(self.params['num_epochs']) +
                         ', train loss: ' + '{:.4f}'.format(round(train_loss.item(), 4)) +
                         ', test_loss: ' + '{:.4f}'.format(round(test_loss, 4)))

    def infer(self, x):

        if len(x.shape) == 2:
            # If x is a single image, unsqueeze twice (once for channel, once for batch)
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)

        elif len(x.shape) == 3:
            # If x is of dim 3, it is assumed that the missing dim is the channel dim.
            x = torch.unsqueeze(x, dim=1)

        preds = torch.exp(self.model(x.to(self.device)))

        return preds

    def __call__(self, x):
        return self.infer(x)

    def plot_history(self, path, save=True, show=False):

        num_epochs = self.params['num_epochs']
        plt.figure()
        plt.plot(np.arange(1, num_epochs + 1), self.history['train_loss'], label='Training loss')
        plt.plot(np.arange(1, num_epochs + 1), self.history['test_loss'], label='Test loss')
        plt.legend(loc='best')
        if save:
            plt.savefig(path + '/classifier_loss.jpg')
        if show:
            plt.show()

        plt.figure()
        plt.plot(np.arange(1, num_epochs + 1), self.history['train_accuracy'], label='Training accuracy')
        plt.plot(np.arange(1, num_epochs + 1), self.history['test_accuracy'], label='Test accuracy')
        plt.legend(loc='best')
        if save:
            plt.savefig(path + 'classifier_accuracy.jpg')
        if show:
            plt.show()

