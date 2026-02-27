"""
Core mathematical functions for curvature bifurcation analysis.
Implements the nonlinear self-modeling function f(θ) and its derivatives.
"""

import numpy as np

def f(theta):
    """
    Nonlinear self-modeling function.
    
    Args:
        theta: numpy array of shape (n,)
        
    Returns:
        f(theta) = tanh(theta) + 0.2 * sin(theta)
    """
    return np.tanh(theta) + 0.2 * np.sin(theta)


def J(theta):
    """
    Jacobian of f with respect to theta.
    
    Args:
        theta: numpy array of shape (n,)
        
    Returns:
        Jacobian matrix of shape (n, n) (diagonal in this simplified case)
    """
    return np.diag(1 - np.tanh(theta)**2 + 0.2 * np.cos(theta))


def H_f(theta):
    """
    List of Hessian matrices for each output component of f.
    
    For this separable function, each Hessian is diagonal.
    
    Args:
        theta: numpy array of shape (n,)
        
    Returns:
        List of n matrices, each of shape (n, n)
    """
    n = len(theta)
    H_list = []
    for i in range(n):
        t = theta[i]
        H_i = np.zeros((n, n))
        # Second derivative of tanh: -2 tanh(t) (1 - tanh²(t))
        # Second derivative of sin: -sin(t)
        H_i[i, i] = -2 * np.tanh(t) * (1 - np.tanh(t)**2) - 0.2 * np.sin(t)
        H_list.append(H_i)
    return H_list


def S(theta):
    """
    Self-consistency Hessian component S(θ) = H_self.
    
    S = (J - I)^T (J - I) + ∑ r_i ∇²f_i
    
    Args:
        theta: numpy array of shape (n,)
        
    Returns:
        S matrix of shape (n, n)
    """
    n = len(theta)
    
    # Residual
    r = f(theta) - theta
    
    # Jacobian
    Jmat = J(theta)
    
    # Linear component (positive semidefinite)
    H_lin = (Jmat - np.eye(n)).T @ (Jmat - np.eye(n))
    
    # Nonlinear component (indefinite)
    H_nl = np.zeros((n, n))
    Hf_list = H_f(theta)
    for i in range(n):
        H_nl += r[i] * Hf_list[i]
    
    return H_lin + H_nl


def create_task_hessian(n, num_classes=5, seed=None):
    """
    Create a realistic task Hessian approximating a softmax classification problem.
    
    H_task = W^T W where W is random (num_classes × n)
    
    Args:
        n: dimension
        num_classes: number of classes
        seed: random seed for reproducibility
        
    Returns:
        Positive definite matrix H_task of shape (n, n)
    """
    if seed is not None:
        np.random.seed(seed)
    
    W = np.random.randn(num_classes, n)
    H_task = W.T @ W
    
    # Ensure symmetry
    H_task = (H_task + H_task.T) / 2
    
    return H_task
