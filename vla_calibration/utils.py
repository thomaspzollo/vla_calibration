import torch
from torch import nn, optim
from torch.nn import functional as F
import pickle as pkl
from scipy.special import softmax
import numpy as np
from pathlib import Path
import calibration as cal
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import entropy
from tqdm import tqdm
import os



def cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss between target values and predictions.

    Args:
    y_true (numpy array): Ground truth labels (1D array, values 0 or 1).
    y_pred (numpy array): Predicted probabilities (1D array, values between 0 and 1).

    Returns:
    float: Binary cross-entropy loss.
    """
    epsilon = 1e-12  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def get_base_data(data_save_dir, top_n_steps=1):

    with open(f"{data_save_dir}/episode_data_true_prompt.pkl", "rb") as f:  # "rb" = read binary mode
        data = pkl.load(f)

    all_probs = []
    all_actions = []
    correct = []

    for episode in data:

        steps = episode["steps"]

        episode_probs = []
        episode_actions = []

        for step in steps[:top_n_steps]:


            logits = step["logits"]

            probs = softmax(logits, -1)

            episode_probs.append(probs)
            episode_actions.append(step["actions"])

        episode_probs = np.stack(episode_probs)
        episode_actions = np.stack(episode_actions)

        all_probs.append(episode_probs)
        all_actions.append(episode_actions)

        correct.append(int(episode["done"]))

    all_probs = np.stack(all_probs)
    all_actions = np.stack(all_actions)
    correct = np.array(correct)

    return all_probs, all_actions, correct



def get_ece(conf, labels, n_bins=15, p=1):
    ece = cal.lower_bound_scaling_ce(conf, labels, p=p, debias=False, num_bins=n_bins, binning_scheme=cal.get_equal_bins, mode='top-label')
    return ece


def sort_a_apply_to_b(a, b):
    """
    Sorts a numpy array `a` and applies the same sorting order to `b`, 
    without modifying the original arrays.

    Parameters
    ----------
    a : np.ndarray
        1D numpy array to be sorted.
    b : np.ndarray
        1D numpy array to be reordered according to `a`.

    Returns
    -------
    a_sorted : np.ndarray
        Sorted version of `a`.
    b_sorted : np.ndarray
        `b` reordered according to the sorted order of `a`.
    """
    # Get sorting indices
    sort_indices = np.argsort(a)[::-1]
    
    # Create sorted copies
    a_sorted = a[sort_indices].copy()
    b_sorted = b[sort_indices].copy()
    
    return a_sorted, b_sorted





def get_scaling_base_data(data_save_dir, top_n_steps=10):

    with open(f"{data_save_dir}/episode_data_true_prompt.pkl", "rb") as f:  # "rb" = read binary mode
        data = pkl.load(f)

    all_probs = []
    all_logits = []
    correct = []

    for episode in data:

        steps = episode["steps"]

        episode_probs = []
        episode_logits = []

        for step in steps[:top_n_steps]:


            logits = step["logits"]

            probs = softmax(logits, -1)

            episode_probs.append(probs)
            episode_logits.append(logits)

        episode_probs = np.stack(episode_probs)
        episode_logits = np.stack(episode_logits)

        all_probs.append(episode_probs)
        all_logits.append(episode_logits)

        correct.append(int(episode["done"]))

    all_probs = np.stack(all_probs)
    all_logits = np.stack(all_logits)
    correct = np.array(correct)

    print("assert", all_logits.shape, all_probs.shape)

    assert all_logits.shape == all_probs.shape

    return all_probs, all_logits, correct


def get_scaling_data(
        task_name, 
        model_type="openvla",
        quant=None,
        alternate_set=1, 
        top_n_steps=1, 
        n_prompts=20, 
        n_cal_bins=12,
):
    
    exp_data_save_str = f"../data/scaling_exp_{task_name}"
    if quant is not None:
        exp_data_save_str += f"_{quant}"
    exp_data_save_str += ".pkl"

    path = Path(exp_data_save_str).expanduser().resolve()
    verbose=True

    if path.is_file():
        if verbose:
            print(f"[load_or_create_pickle] Loading existing pickle: {path}")
        with path.open("rb") as f:
            output_data = pkl.load(f)
        
    else:

        print("Building data...", exp_data_save_str)
        
        data_save_dir = f"../results/libero_{task_name}"
        if quant is not None:
            data_save_dir += f"/{quant}"

        base_probs, base_logits, correct = get_scaling_base_data(data_save_dir, top_n_steps)

        base_probs = np.expand_dims(base_probs, axis=2)
        base_logits = np.expand_dims(base_logits, axis=2)

        assert base_probs.shape == base_logits.shape
        
        print("Data shapes:")
        print("Probs:", base_probs.shape)
        print("Logits", base_logits.shape)
        print("Correctness", correct.shape)
        print()
        print("Accuracy:", np.mean(correct))

        by_dim_results = {"baseline":[],"ensemble":[]}

        x = base_probs[:,0,0]
        x = np.max(x, -1)

        # print("--------\nBaseline")
        for i in range(x.shape[-1]):
            dim_ece = get_ece(x[:,i], correct, n_cal_bins)
            # print(f"Dimension {i} ECE: {dim_ece}")
            by_dim_results["baseline"].append(dim_ece)
        all_ece = get_ece(np.mean(x, -1), correct, n_cal_bins)
        

        all_probs = []
        all_logits = []

        for i in range(n_prompts):

            prompt_probs = []
            prompt_logits = []

            if alternate_set == 1:
                data_save_str = f"{data_save_dir}/episode_data_prompt_{i}.pkl"
            elif alternate_set == 2:
                data_save_str = f"{data_save_dir}/episode_data_prompt_{i}_v2.pkl"
            elif alternate_set == 3:
                data_save_str = f"{data_save_dir}/episode_data_prompt_{i}_v3.pkl"
            else:
                raise ValueError

            with open(data_save_str, "rb") as f:  # "rb" = read binary mode
                data = pkl.load(f)

            for episode in data:

                episode_probs = []
                episode_logits = []

                steps = episode["steps"]

                for step in steps[:top_n_steps]:

                    logits = step["logits"]
                    probs = softmax(logits, -1)

                    episode_probs.append(probs)
                    episode_logits.append(logits)
                    

                episode_probs = np.stack(episode_probs)
                episode_logits = np.stack(episode_logits)
                

                prompt_probs.append(episode_probs)
                prompt_logits.append(episode_logits)
                

            prompt_probs = np.stack(prompt_probs)
            prompt_logits = np.stack(prompt_logits)
            

            all_probs.append(prompt_probs)
            all_logits.append(prompt_logits)
            
        all_probs = np.stack(all_probs)
        all_probs = np.transpose(all_probs, (1,2,0,3,4))

        all_logits = np.stack(all_logits)
        all_logits = np.transpose(all_logits, (1,2,0,3,4))

        assert all_probs.shape == all_logits.shape

        ens_probs = all_probs
        ens_logits = all_logits

        x = all_probs[:,0]
        x = np.max(x, -1)
        x = np.mean(x, 1)
        
        for i in range(x.shape[-1]):
            dim_ece = get_ece(x[:,i], correct, n_cal_bins)
            
            by_dim_results["ensemble"].append(dim_ece)
        all_ece = get_ece(np.mean(x, -1), correct, n_cal_bins)

        output_data = (base_probs, ens_probs, base_logits, ens_logits, correct, by_dim_results)


        with open(exp_data_save_str, "wb") as f:
            pkl.dump(output_data, f)

    return output_data




def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits data matrix X and label vector y into training and testing sets.
    
    Parameters:
    - X: np.ndarray, shape (m, n), the data matrix.
    - y: np.ndarray, shape (m,), the labels.
    - test_size: float, fraction of data to be used as test set (default is 0.2).
    - random_state: int or None, seed for reproducibility.
    
    Returns:
    - X_train: np.ndarray, training data.
    - X_test: np.ndarray, testing data.
    - y_train: np.ndarray, training labels.
    - y_test: np.ndarray, testing labels.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    test_count = int(np.floor(test_size * m))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def train_test_split_three_way(X, z, y, test_size=0.2, random_state=42):
    """
    Splits data matrix X and label vector y into training and testing sets.
    
    Parameters:
    - X: np.ndarray, shape (m, n), the data matrix.
    - y: np.ndarray, shape (m,), the labels.
    - test_size: float, fraction of data to be used as test set (default is 0.2).
    - random_state: int or None, seed for reproducibility.
    
    Returns:
    - X_train: np.ndarray, training data.
    - X_test: np.ndarray, testing data.
    - y_train: np.ndarray, training labels.
    - y_test: np.ndarray, testing labels.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    test_count = int(np.floor(test_size * m))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    z_train, z_test = z[train_indices], z[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, z_train, z_test, y_train, y_test


