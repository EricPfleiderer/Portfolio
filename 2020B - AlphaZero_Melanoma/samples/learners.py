import numpy as np
import random
import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Input, Concatenate, BatchNormalization, Reshape
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import Model
from keras.utils import plot_model
from keras.backend import zeros, shape, equal

from AlphaSilico.src.MCTS import Node, Edge, MCTS
from AlphaSilico.src.insilico import State
from AlphaSilico.src import config
from scipy.integrate import ode, cumtrapz


class Losses:

    @staticmethod
    def softmax_cross_entropy_with_logits(y_true, y_pred):

        pi = y_true
        p = y_pred

        zero = zeros(shape=shape(pi), dtype=tf.float32)
        where = equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0)
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

        return loss


class Learner:

    def __init__(self, learning_rate, input_shapes=((10,), (28,)), policy_size=5, value_size=1):
        """
        Multiple input, twin output model.
        :param learning_rate: Float. Learning rate during training.
        :param input_shapes: Tuple of tuples. Each entry represents the input shape of a head.
        :param policy_size: Int. Policy head output shape.
        :param value_size: Int. Value head output shape.
        """
        self.learning_rate = learning_rate
        self.input_shapes = input_shapes
        self.policy_size = policy_size
        self.value_size = value_size
        self.model = self._build_model()

    def _core(self, merged_input):
        x = Dense(50, kernel_regularizer=l2())(merged_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(25, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def _value_head(self, x):
        x = Dense(10, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        value_head = Dense(self.value_size, use_bias=False, activation='tanh', kernel_regularizer=l2(), name='value_head')(x)
        return value_head

    def _policy_head(self, x):
        x = Dense(10, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(self.policy_size**2, use_bias=False, activation='linear', kernel_regularizer=l2())(x)
        policy_head = Reshape((self.policy_size, self.policy_size), name='policy_head')(x)
        return policy_head

    def _build_model(self):

        # Accept and merge inputs
        inputs = [Input(shape=input_shape) for input_shape in self.input_shapes]
        merged_input = Concatenate()(inputs)

        # Build the core from the merged input
        core = self._core(merged_input)

        # Build the heads from the core
        value_head = self._value_head(core)
        policy_head = self._policy_head(core)

        # Initialize and compile model
        model = Model(inputs=inputs, outputs=[value_head, policy_head])
        model.compile(loss={'value_head': 'mean_squared_error',
                            'policy_head': Losses.softmax_cross_entropy_with_logits
                            },
                      optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})

        # Return the final model
        return model

    def convert_to_input(self, state):
        pass

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split, batch_size=batch_size)

    def write(self, model, version):
        pass

    def read(self, model, version):
        pass

    def plot_model(self):
        if self.model is not None:
            plot_model(self.model, to_file='outputs/Model_graph.png')

    @staticmethod
    def convert_to_model_input(state):
        params_array = [state.variable_params]
        y_array = [state.y]
        return [params_array, y_array]


class Agent:

    def __init__(self, name, action_size, mcts_simulations, cpuct, learner):

        """
        Self-reinforcement learning agent. Uses twin headed model and Monte Carlo tree searches according to AlphaZero architecture by deepmind.
        Interfaces with State class.
        :param name: String. Agent designation.
        :param action_size: Int. Maximum allowed dose.
        :param mcts_simulations: Int. Number of MCTS simulations.
        :param cpuct: Float. Exploration constant.
        :param learner: Learner class. Interface to neural network.
        """
        self.name = name
        self.action_size = action_size
        self.cpuct = cpuct
        self.mcts_simulations = mcts_simulations
        self.brain = learner

        self.mcts = None
        self.root = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def act(self, state, after_tau0, tau):

        # Build a new tree if the state is not previously visited
        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_MCTS_root(state)

        # Change the root of the tree to the current state otherwise
        else:
            self.change_MCTS_root(state)

        # Fill the Monte Carlo tree
        for sim in range(self.mcts_simulations):
            self.simulate()

        # Get Monte Carlo statistics
        pi, values = self.get_action_values(tau=tau)

        # Pick the action to play
        action, value = self.choose_action(pi, values, after_tau0=after_tau0)

        # Play the action and get the output of value head for the next state
        next_state, _, _ = state.take_action(action)
        NN_value, _, _ = self.get_preds(next_state)

        return action, pi, value, NN_value

    def build_MCTS_root(self, state):
        """
        Builds a new Monte Carlo tree (root only).
        :param state: State instance. Root node for the new tree.
        :return: Void.
        """
        self.root = Node(state)
        self.mcts = MCTS(self.root, self.cpuct)

    def change_MCTS_root(self, state):
        self.mcts.root = self.mcts.tree[state.id]

    @staticmethod
    def choose_action(pi, values, after_tau0):

        # Deterministic play
        if after_tau0:
            actions = np.argwhere(pi == np.max(pi))
            random_idx = np.random.randint(0, actions.shape[0])  # Select random maximum if more than one exists
            action = (actions[random_idx][0], actions[random_idx][1])

        # Random play
        else:
            flat_pi = pi.flatten()
            action_idx = np.random.multinomial(1, flat_pi).reshape(pi.shape)
            action = (np.where(action_idx == 1)[0][0], np.where(action_idx == 1)[1][0])

        value = values[action]

        return action, value

    def evaluate_expand(self, leaf, value, done, breadcrumbs):

        """
        If node is not terminal, evaluate current node and expand. Else, return
        :param leaf: Node instance. Evaluate the leaf. If the simulation is not done, expand it as well.
        :param value: Float. Position estimation given by value head.
        :param done: Boolean. True if the node is a leaf.
        :param breadcrumbs: List of Edge objects. Ordered list of visited edges during a call to select().
        :return:
        """

        if not done:
            value, probs, legal_actions = self.get_preds(leaf.state)  # Value head, policy head, legal actions

            # Expand current leaf through all legal actions
            for idx, action in enumerate(legal_actions):
                action = tuple(action)  # Prepare for use as indice
                new_state, _, _ = leaf.state.take_action(action)

                # If the state has not previously been visited, add it to the tree
                if new_state.id not in self.mcts.tree:
                    node = Node(new_state)
                    self.mcts.add_node(node)

                else:  # SLOW!!! (recomputing previously visited edges is expensive) MUST REFACTOR IDs, why would we ever enter this condition..? bug..
                    node = self.mcts.tree[new_state.id]

                # Add newly expanded edge to the leaf
                new_edge = Edge(leaf, node, probs[action], action)
                leaf.edges.append((action, new_edge))

        return value, breadcrumbs

    def get_action_values(self, tau):

        """
        Get values of each move and their probability to be played according to MCTS simulation.
        :param tau: Int. Turns before deterministic play.
        :return: 2D numpy arrays. Values and probabilities.
        """

        edges = self.mcts.root.edges
        pi = np.zeros((config.MAX_DOSES+1, config.MAX_DOSES+1), dtype=np.integer)
        values = np.zeros((config.MAX_DOSES+1, config.MAX_DOSES+1), dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi))  # Convert to probability (SOFTMAX)

        return pi, values

    def get_preds(self, state):
        """
        Get model predictions under allowed actions constraints
        :param state: State instance. Current state to predict.
        :return: Float, ndarray, ndarray. Predicted state value, probability map, legal actions.
        """

        # Predict the leaf
        model_input = self.brain.convert_to_model_input(state)
        preds = self.brain.predict(model_input)
        value = preds[0][0][0]  # Float value (axis1=output, axis2=example, axis3=feature)
        policy = preds[1][0]

        # Convert prediction to probability (SOFTMAX)model_input
        legal_meshgrid, legal_actions = state.get_available_actions(max_dose=4)  # Mask the illegal moves
        legal_policy = np.ones(shape=policy.shape) * -9999  # Squashing
        legal_policy[legal_meshgrid] = policy[legal_meshgrid]
        z = legal_policy-np.max(legal_policy)  # Avoid underflows by substracting max value
        probs = np.exp(z)
        probs /= np.sum(probs)

        return value, probs, legal_actions

    def predict(self, input_to_model):
        return self.brain.predict(input_to_model)

    def replay(self, lt_memory):
        """
        Fits the Agent to stored data.
        :param lt_memory:
        :return:
        """

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(lt_memory, min(config.BATCH_SIZE, len(lt_memory)))

            # Build training examples
            training_states = [[], []]
            for example in minibatch:
                training_states[0].append(example['params'])
                training_states[1].append(example['y'])

            # Build training targets
            training_targets = {'value_head': np.array([example['value'] for example in minibatch]),
                                'policy_head': np.array([example['pi'] for example in minibatch])}

            fit = self.brain.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

    def simulate(self):

        """
        Agent look-ahead using MCTS as per AlphaZero architecture. Repeated calls will fill the MC tree.
        :return: Void.
        """

        # Select a leaf.
        leaf, value, done, breadcrumbs = self.mcts.select()  # Value == Y_true

        # Evaluate and expand if selected node is terminal.
        value, breadcrumbs = self.evaluate_expand(leaf, value, done, breadcrumbs)  # Value == Y_pred

        # Backup the value through the breadcrumbs
        self.mcts.backup(value, breadcrumbs)

