import torch
from src.trainable import TorchTrainable
import matplotlib.pyplot as plt
import numpy as np


# Attacker algorithm here
class GeneticAttack:

    def __init__(self, x: torch.Tensor, y: int, trainable: TorchTrainable, N: int = 10, epochs=50,
                 selective_pressure: float = 0.2, mutation_size=0.025, asexual_repro: float = 1, epsilon: float = 0.1,
                 uncertainty_power: int = 2, sameness_power: int = 4):

        """
        :param x: 28 by 28 torch tensor of the original image.
        :param y: Target.
        :param trainable: Trainable targeted by the attack.
        :param N; Size of the population during the simulation. Strongly recommended to use a pair integer.
        :param epochs: Number of epochs to run the genetic algorithm.
        :param selective_pressure: Percentage of the most fit population that are considered during reproduction.
        :param asexual_repro: Percentage of the population that will reproduce asexually (cloning)
        :param epsilon: Coefficient for the linear combination between uncertainty and sameness in the loss function.
        :param uncertainty_power: Power of the exponent used in the uncertainty term of the loss function.
        :param sameness_power: Power of the exponent used in the sameness term in the loss function.
        """
        self.trainable = trainable
        self.x = x.to(self.trainable.device)
        self.y = y
        self.N = N
        self.epochs = epochs
        self.selective_pressure = selective_pressure
        self.mutation_size = mutation_size
        self.asexual_repro = asexual_repro
        self.epsilon = epsilon
        self.uncertainty_power = uncertainty_power
        self.sameness_power = sameness_power
        self.best_solution = x

        self.history = {
            'uncertainty_loss': np.array([]),
            'sameness_loss': np.array([]),
            'best_solution': np.empty(shape=(0, 28, 28)),
            'prediction_dist': np.empty(shape=(0, 10)),
        }

        # Create a population by duplicating the attack target (x)
        population = torch.stack([x for i in range(N)]).to(self.trainable.device)

        # Run a genetic attack
        for i in range(epochs):

            # TODO: add different types of mutations
            # TODO: implement perturbation decay

            # Evaluate the quality of the population
            quality = self.evaluate_quality(population)

            # Rank the population in order of descending quality (loss minimization)
            rank = torch.argsort(quality, descending=True)

            if i % 10:
                print(f'{i}: {quality[rank[-1]].data}')

            # Choose the fittest units for reproduction (N//2 parents chosen with replacement among the fittest)
            parents_idx = []
            for n in range(self.N // 2):
                parents = self.select_parents(rank)
                parents_idx.append(parents)
            parents_idx = torch.stack(parents_idx)

            # Create the new generation from the fittest parents
            children = self.generate_children(population, parents_idx)

            # Perturb the population with random mutations
            children[torch.where(children != 0)] += torch.normal(0, self.mutation_size, size=children[torch.where(children != 0)].shape).to(self.trainable.device)
            children = torch.clamp(children, 0, 1)

            # Elitism (maintain top solution at all times)
            self.best_solution = population[rank[-1]]
            population = children
            population[0] = self.best_solution

            # Add to history
            uncertainty_loss, sameness_loss = self.loss(self.best_solution)
            self.history['uncertainty_loss'] = np.append(self.history['uncertainty_loss'], uncertainty_loss.cpu().detach().numpy())
            self.history['sameness_loss'] = np.append(self.history['sameness_loss'], sameness_loss.cpu().detach().numpy())
            self.history['best_solution'] = np.concatenate((self.history['best_solution'], np.expand_dims(self.best_solution.cpu().detach().numpy(), axis=0)))
            self.history['prediction_dist'] = np.concatenate((self.history['prediction_dist'], self.trainable(self.best_solution).cpu().detach().numpy()), axis=0)

    def complement_idx(self, idx: torch.Tensor, dim: int):

        """
        Compute the following complement: set(range(dim)) - set(idx).
        SOURCE:  https://stackoverflow.com/questions/67157893/pytorch-indexing-select-complement-of-indices

        :param idx: indexes to be excluded in order to form the complement
        :param dim: max index for the complement
        """
        a = torch.arange(dim, device=idx.device)
        ndim = idx.ndim
        dims = idx.shape
        n_idx = dims[-1]
        dims = dims[:-1] + (-1,)
        for i in range(1, ndim):
            a = a.unsqueeze(0)
        a = a.expand(*dims)
        masked = torch.scatter(a, -1, idx, 0)
        compl, _ = torch.sort(masked, dim=-1, descending=False)
        compl = compl.permute(-1, *tuple(range(ndim - 1)))
        compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
        return compl

    def generate_children(self, population, parents_idx):

        parent_pool = torch.clone(parents_idx)

        # Asexual reproduction (cloning)
        repro_mask = torch.bernoulli(torch.Tensor([self.asexual_repro for n in range(self.N//2)])).to(self.trainable.device)
        mask_idx = torch.where(repro_mask == 1)[0].to(self.trainable.device)  # Apply mask and find parents to clone
        clones = population[parents_idx[mask_idx]]
        clones = torch.flatten(clones, start_dim=0, end_dim=1)

        parent_pool = parent_pool[self.complement_idx(mask_idx, dim=self.N//2)]

        # Reshape the parents index tensor in preparation for ''fancy'' indexing
        inv_parents_idx = parent_pool.resize(parent_pool.shape[1], parent_pool.shape[0])

        # Sexual reproduction (gene sharing)
        r = torch.rand(size=self.x.shape).to(self.trainable.device)
        batch1 = r * population[inv_parents_idx[0]] + (1 - r) * population[inv_parents_idx[1]]
        batch2 = r * population[inv_parents_idx[1]] + (1 - r) * population[inv_parents_idx[0]]
        children = torch.cat([batch1, batch2])
        children = torch.cat([clones, children])

        return children

    def select_parents(self, rank: torch.Tensor):

        """
        :param rank: The descending ranking of the population in terms of quality.
        :return: The index of 2 individuals chosen to become parents.
        """

        # TODO: optimize (naive implementation)

        # We select the first
        lower_bound = int((1 - self.selective_pressure) * self.N)
        first_index = torch.randint(low=lower_bound, high=self.N, size=(1,), device=self.trainable.device)

        # Choose best parent randomly according to selective pressure
        first_parent = rank[first_index]

        # Choose second parent
        second_parent = first_parent
        while second_parent == first_parent:
            second_parent = torch.randint(0, self.N - 1, size=(1,), device=self.trainable.device)

        return torch.tensor([first_parent, second_parent], device=self.trainable.device)

    def evaluate_quality(self, adversarial_x: torch.Tensor):

        """
        :param adversarial_x: batch of 28 by 28 perturbed images
        :return:
        """
        # TODO: Find optimal parameters for quality eval (epsilon, powers)

        uncertainty_loss, sameness_loss = self.loss(adversarial_x)

        return uncertainty_loss + self.epsilon * sameness_loss

    def loss(self, adversarial_x: torch.Tensor):
        uncertainty_loss = self.trainable(adversarial_x)[
                           :, self.y]

        sameness_loss = (self.x-adversarial_x)**self.sameness_power

        if len(adversarial_x.shape) == 2:
            sameness_loss = torch.sum((self.x-adversarial_x)**self.sameness_power).to(self.trainable.device)

        else:
            for x in range(len(adversarial_x.shape)-1):
                sameness_loss = torch.sum(sameness_loss, dim=1)

        sameness_loss = sameness_loss.to(self.trainable.device)

        return uncertainty_loss, sameness_loss

    def plot_history(self, path, save=True, show=False):

        # Initial image
        plt.figure()
        plt.imshow(self.x.detach().cpu())
        if save:
            plt.savefig(path + 'original_x.png')
        if show:
            plt.show()

        # Adversarial image
        plt.figure()
        plt.imshow(self.best_solution.detach().cpu())
        if save:
            plt.savefig(path + 'adversarial_x.png')
        if show:
            plt.show()

        x = range(0, self.epochs)
        plt.figure()
        plt.plot(x, self.history['uncertainty_loss'], label='Uncertainty loss', linestyle='--')
        plt.plot(x, self.epsilon * np.array(self.history['sameness_loss']), label='$\epsilon $' + ' x Sameness loss',
                 linestyle='--')
        plt.plot(x, np.array(self.history['uncertainty_loss']) + self.epsilon * np.array(self.history['sameness_loss']),
                 label='Total loss')
        plt.legend()
        if save:
            plt.savefig(path + 'ga_loss.png')
        if show:
            plt.show()
