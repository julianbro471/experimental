import random

import numpy as np
import data_management as dm
from mathematic_functions import *
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def enable_logging(path: str | Path | None = None, level: int = logging.INFO) -> None:
    """Turns on logging to a file (default: training.log next to this module)."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return  # already enabled
    path = Path(path) if path else Path(__file__).parent / "training.log"
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)

if os.environ.get("POLSKA_LOG") == "1":
    enable_logging()


class polska_gurom(object):
    def __init__(self , sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])] #new and improved weight initialization that is still yet for testing.


    def feedforward(self, x:np.ndarray) -> np.ndarray:
        """
        Computes the feedforward pass of the neural network for a single input vector.

        Parameters:
        x (numpy.ndarray): The input column vector of shape (sizes[0], 1).

        Returns:
        numpy.ndarray: The output vector of the network from the final layer.

        Raises:
        ValueError: If the input 'x' does not match the expected input layer dimensions.
        """
        if x.shape != (self.sizes[0], 1):
            raise ValueError(f"Input vector 'x' must have shape ({self.sizes[0]}, 1), but got {x.shape}")

        input = x
        for i in range(len(self.weights) - 1):
            w = self.weights[i]
            b = self.biases[i]
            z = (w @ input) + b
            input = sigmoid_function(z)

        return softmax_function(self.weights[-1] @ input + self.biases[-1]) if self.sizes[-1] > 1 else input
        # return input

    def result_test_data(self, data) -> int:
        """
        Evaluates the neural network on a test dataset and returns the number of correct predictions.

        Parameters:
        data (iterable): An iterable collection of tuples (x, y) where 'x' is the input vector 
                         and 'y' is the true label/target value.

        Returns:
        int: The total count of correct predictions.

        Raises:
        ValueError: If the provided dataset is empty or None.
        RuntimeError: If an error occurs during inference or sample processing.
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty.")
            
        counter = 0
        is_binary = self.sizes[-1] == 1
        
        for x, y in data:
            try:
                output = self.feedforward(x)
                if is_binary:
                    prediction = 1 if output[0, 0] >= 0.5 else 0
                else:
                    prediction = np.argmax(output)

                if isinstance(y, np.ndarray):
                    y = np.argmax(y) 
                if prediction == y:
                    counter += 1
            except Exception as e:
                raise RuntimeError(f"Error processing test sample: {e}")
                
        return counter

    def confusion_matrix(self, data) -> np.ndarray:
        """
        Builds the confusion matrix of the network on a dataset.

        Parameters:
        data (iterable): An iterable collection of tuples (x, y).

        Returns:
        np.ndarray: Matrix of shape (n_classes, n_classes). Rows are true classes, columns are predicted classes.

        Raises:
        ValueError: If the provided dataset is empty or None.
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty.")

        is_binary = self.sizes[-1] == 1
        n_classes = 2 if is_binary else self.sizes[-1]
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for x, y in data:
            output = self.feedforward(x)
            if is_binary:
                prediction = 1 if output[0, 0] >= 0.5 else 0
            else:
                prediction = int(np.argmax(output))
            if isinstance(y, np.ndarray):
                y = int(np.argmax(y)) if y.size > 1 else int(y.item())
            matrix[y, prediction] += 1

        return matrix

    def evaluation_report(self, data):
        """
        Prints the confusion matrix with accuracy, precision, recall, F1, and support.

        Precision = correct predictions of column sum / all
        Recall = correct predictions of row sum / all
        F1 = 2 * precision * recall / (precision + recall)

        Parameters:
        data (iterable): An iterable collection of tuples (x, y).
        """
        matrix = self.confusion_matrix(data)
        correct = np.diag(matrix)
        predicted = matrix.sum(axis=0)
        actual = matrix.sum(axis=1)

        accuracy = correct.sum() / matrix.sum()
        precision = np.divide(correct, predicted, out=np.zeros(len(correct)), where=predicted > 0)
        recall = np.divide(correct, actual, out=np.zeros(len(correct)), where=actual > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros(len(correct)), where=denom > 0)

        n = len(matrix)
        print("Confusion matrix:")
        print(f'{"tr\\pr":<6}' + "".join(f"{j:<6}" for j in range(n)))
        for i in range(n):
            print(f"{i:<6}" + "".join(f"{v:<6}" for v in matrix[i]))
        print(f"\nAccuracy: {correct.sum()} / {matrix.sum()} = {accuracy:.4f}\n")
        print(f"{'class':<6}{'precision':<11}{'recall':<9}{'f1':<8}{'support':<9}")
        for i in range(n):
            print(f"{i:<6}{precision[i]:<11.4f}{recall[i]:<9.4f}{f1[i]:<8.4f}{actual[i]:<9}")

    def update_mini_batch(self, mini_batch: list, eta: float, lmbda: float = 0.0, n: int = 1) -> tuple:
        """
        Updates the network's weights and biases by applying gradient descent
        using backpropagation to a single mini-batch.

        Parameters:
        mini_batch (list): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
        eta (float): The learning rate.
        lmbda (float): The regularization parameter.
        n (int): The total number of training examples.

        Returns:
        tuple: Updated weights and biases after applying gradient descent.
        """

        # Initialize gradients for weights and biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Perform backpropagation for each example in the mini-batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        # Update weights and biases using the average gradients
        self.weights = [w * (1 - eta * lmbda / n) - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    
    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Parameters:
        x (np.ndarray) - An array of the activations for the input of the network
        y (np.ndarray) - An array with outputs for x
        
        Returns: 
        tuple: A tuple containing two lists of np.ndarray, the gradients.
        
        Raises:
        ValueError: If the input 'x' or 'y' does not match the expected input layer dimensions.
        """
        
        if x.shape != (self.sizes[0], 1):
            raise ValueError(f"Input 'x' must have shape ({self.sizes[0]}, 1), got {x.shape}")
        if y.shape != (self.sizes[-1], 1):
            raise ValueError(f"Target 'y' must have shape ({self.sizes[-1]}, 1), got {y.shape}")
        
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        
        input = x 
        activations = [x]
        zs = []
        
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, input) + b
            zs.append(z)
            input = sigmoid_function(z)
            activations.append(input)

        z = np.dot(self.weights[-1], input) + self.biases[-1]
        zs.append(z)
        activations.append(softmax_function(z))
        
        error = activations[-1] - y
        grad_b[-1] = error
        grad_w[-1] = np.dot(error, activations[-2].T)
        
        for l in range(2, self.num_layers):
            error = np.dot(self.weights[-l+1].T, error) * sigmoid_derivative(sigmoid_function(zs[-l]))
            grad_b[-l] = error
            grad_w[-l] = np.dot(error, activations[-l-1].T)
        
        return (grad_b, grad_w)

    def train(self, training_data: list[tuple[np.ndarray,np.ndarray]], epochs: int, batch_size: int, learning_rate: float, validation_data: list[tuple[np.ndarray, int]] = None,
              patience: int = None):
        """

        Parameters:
            training_data (list[tuple[np.ndarray,np.ndarray]]): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
            epochs (int): Number of epochs for how long the training must go.
            batch_size (int): Size of the mini batches.
            learning_rate (float): Learning rate
            test_data (list[tuple[np.ndarray, int]], optional): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
            It is used to compare the training results with the correct ones. Using it slows down the algorithm. Defaults to None.
        
        Raises:
            ValueError: If the input 'training_data' is empty.
            ValueError: If the number of epochs is less or equal than 0.
            ValueError: If the value of the learning rate is less than zero.
        """
        
        if not training_data:
            raise ValueError("Training Data shouldn't be empty")
        if epochs <= 0:
            raise ValueError(f"Epochs must be an integer greater than 0, got {epochs}")
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be greater than 0, got {learning_rate}")
        if patience is not None and patience <= 0:
            raise ValueError(f"Patience must be greater than 0, got {patience}")
        
        if validation_data:
            n_validation = len(validation_data)
        n = len(training_data)

        best_score = -1
        best_epoch = -1
        best_weights = None
        best_biases = None
        
        
        for i in range(epochs):
            
            batches = batch_split(training_data, batch_size)    
            for batch in batches:
                self.update_mini_batch(batch, learning_rate, n = n)

            score = self.result_test_data(validation_data) if validation_data else None
            if validation_data:
                print(f"Epoch {i}: {score} / {n_validation}")
            else:
                print(f"Epoch {i} completed")

            train_score = self.result_test_data(random.sample(training_data, 2000))
            if train_score > score: logger.info(f"Epoch {i}: train {train_score}/2000, validation {score}/{n_validation}")

            if score is not None and score > best_score:
                best_score = score
                best_epoch = i
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            elif patience is not None and i - best_epoch >= patience:
                print(f"No improvement for {patience} epochs, stopping at epoch {i}.")
                break

        # if patience is not None and best_weights is not None:
            #TODO: save the best weight and scores to a json file.

def batch_split(training_data: list, mini_batch_size: int = 10) -> list:
        """
        Shuffles the training data and splits it into mini-batches.

        Parameters:
        training_data (list): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
        mini_batch_size (int): Number of examples per mini-batch (the last one may be smaller).

        Returns:
        list: A list of mini-batches, each a list of (x, y) tuples.
        """

        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be positive")

        data = list(training_data)
        random.shuffle(data)

        return [data[k:k + mini_batch_size] for k in range(0, len(data), mini_batch_size)]

def load_data(pictures: str, labels: str) -> list[tuple[np.ndarray, int]]:
    """
    Converts pictures and labels into a format suitable for the neural network.

    Parameters:
    pictures (str): Path to the folder containing the pictures.
    labels (str): Path to the file containing the labels.

    Returns:
    list[tuple[np.ndarray, int]]: A list of tuples (x, y) where 'x' is the input vector and 'y' is the label.

    Raises:
    ValueError: If the lengths of pictures and labels do not match.
    """
    pictures = dm.pictures_load(pictures)
    labels = dm.load_labels(labels)

    if len(pictures) != len(labels):
        raise ValueError("The number of pictures must match the number of labels")

    data = []
    for x, y in zip(pictures, labels):
        vector = np.zeros((10, 1))
        vector[y] = 1.0
        data.append((x, vector))
    return data