import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm




class TempScaler(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(TempScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)


    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.view(1,1,1)
        out = logits / temperature
        out = out.softmax(-1)
        out = torch.max(out, -1)[0]
        out = torch.mean(out, -1)
        return out



    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.BCELoss().cuda()

        logits = logits.cuda()
        labels = labels.cuda()

        scaled_logits = self.temperature_scale(logits)

        before_temperature_nll = nll_criterion(scaled_logits, labels).item()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()

        return self


class PlattScaler:
    def __init__(self, max_iter=100, tol=1e-5):
        """
        Initializes the PlattScaler.
        
        Parameters:
        - max_iter: Maximum number of iterations for the optimizer.
        - tol: Tolerance for the convergence criterion.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.A = None
        self.B = None

    def _sigmoid(self, x):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def fit(self, probabilities, correctness):
        """
        Fit the Platt scaling model.
        
        Parameters:
        - probabilities: list or array of unscaled probabilities (or scores), shape (n,).
        - correctness: list or array of binary outcomes (0 or 1) indicating correctness, shape (n,).
        
        Returns:
        - self: fitted PlattScaler instance.
        """
        X = np.array(probabilities)
        y = np.array(correctness)
        
        def objective(params):
            A, B = params
            p = self._sigmoid(A * X + B)
            eps = 1e-6
            p = np.clip(p, eps, 1 - eps)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        def grad(params):
            A, B = params
            p = self._sigmoid(A * X + B)
            error = p - y
            grad_A = np.sum(error * X)
            grad_B = np.sum(error)
            return np.array([grad_A, grad_B])
        
        result = minimize(
            objective,
            x0=[1.0, 0.0],
            jac=grad,
            method='BFGS',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        self.A, self.B = result.x
        return self

    def predict(self, probabilities):
        """
        Predict calibrated probabilities.
        
        Parameters:
        - probabilities: list or array of unscaled probabilities, shape (n,).
        
        Returns:
        - calibrated: numpy array of calibrated probabilities, shape (n,).
        """
        if self.A is None or self.B is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        
        X = np.array(probabilities)
        calibrated = self._sigmoid(self.A * X + self.B)
        return calibrated

class ActionPlattScaler:
    def __init__(self, max_iter=100, tol=1e-5, combine_method="product"):
        """
        Initializes the ActionPlattScaler.
        
        Parameters:
        - max_iter: Maximum number of iterations for each underlying PlattScaler.
        - tol: Tolerance for convergence of the optimizer.
        - combine_method: How to combine calibrated probabilities from each dimension.
                          Options include "product" (default), "mean", or "weighted".
                          "weighted" learns a logistic regression combiner.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.combine_method = combine_method
        self.scalers = []  # One PlattScaler per action dimension
        self.combiner = None  # Parameters for logistic regression combiner, if used

    def fit(self, action_probabilities, correctness):
        """
        Fit the ActionPlattScaler on a matrix of raw probabilities and a single correctness vector.
        
        Parameters:
        - action_probabilities: m×a matrix where each column corresponds to a different action dimension.
        - correctness: m-dimensional vector of binary outcomes (0 or 1) indicating overall correctness.
        
        Returns:
        - self: fitted ActionPlattScaler instance.
        """
        X = np.array(action_probabilities)
        y = np.array(correctness)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of examples in action_probabilities and correctness must match.")
        
        num_actions = X.shape[1]
        self.scalers = []
        
        # Train a separate PlattScaler for each action dimension using the same correctness labels.
        for i in range(num_actions):
            scaler = PlattScaler(max_iter=self.max_iter, tol=self.tol)
            scaler.fit(X[:, i], y)
            self.scalers.append(scaler)
        
        # If using the weighted combination, train a logistic regression combiner.
        if self.combine_method == "weighted":
            # First, compute calibrated probabilities for each action dimension on training data.
            calibrated_features = np.zeros_like(X, dtype=float)
            for i, scaler in enumerate(self.scalers):
                calibrated_features[:, i] = scaler.predict(X[:, i])
            
            # Define the logistic regression model on the calibrated features.
            # Model: sigma(w^T f + b), where f is the calibrated feature vector.
            def combiner_objective(params):
                w = params[:-1]
                b = params[-1]
                logits = np.dot(calibrated_features, w) + b
                p = 1 / (1 + np.exp(-logits))
                eps = 1e-6
                p = np.clip(p, eps, 1 - eps)
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            
            def combiner_grad(params):
                w = params[:-1]
                b = params[-1]
                logits = np.dot(calibrated_features, w) + b
                p = 1 / (1 + np.exp(-logits))
                error = p - y
                grad_w = np.dot(calibrated_features.T, error)
                grad_b = np.sum(error)
                return np.concatenate([grad_w, [grad_b]])
            
            # Initialize combiner parameters: one weight per dimension and one bias.
            init_params = np.zeros(num_actions + 1)
            result = minimize(
                combiner_objective,
                x0=init_params,
                jac=combiner_grad,
                method='BFGS',
                options={'maxiter': self.max_iter, 'gtol': self.tol}
            )
            self.combiner = result.x
        
        return self

    def predict(self, action_probabilities):
        """
        Predict a combined calibrated probability for each example.
        
        Parameters:
        - action_probabilities: s×a matrix of raw probabilities for each action dimension.
        
        Returns:
        - combined: numpy array of combined calibrated probabilities, shape (s,).
        """
        X = np.array(action_probabilities)
        s, num_actions = X.shape
        
        if num_actions != len(self.scalers):
            raise ValueError("Mismatch between number of action dimensions and trained scalers.")
        
        # Calibrate each action dimension independently.
        calibrated = np.zeros((s, num_actions))
        for i, scaler in enumerate(self.scalers):
            calibrated[:, i] = scaler.predict(X[:, i])
        
        if self.combine_method == "product":
            combined = np.prod(calibrated, axis=1)
        elif self.combine_method == "mean":
            combined = np.mean(calibrated, axis=1)
        elif self.combine_method == "weighted":
            if self.combiner is None:
                raise ValueError("Combiner has not been trained. Call 'fit' first.")
            w = self.combiner[:-1]
            b = self.combiner[-1]
            logits = np.dot(calibrated, w) + b
            combined = 1 / (1 + np.exp(-logits))
        else:
            raise ValueError("Unsupported combine_method. Use 'product', 'mean', or 'weighted'.")
        
        return combined

