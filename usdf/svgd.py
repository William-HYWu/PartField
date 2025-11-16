import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SVGD:
    """
    Implementation of Stein Variational Gradient Descent (SVGD).
    
    Reference:
    [1] Liu, Q., & Wang, D. (2016). Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm.
        Advances in Neural Information Processing Systems.
    """
    def __init__(self):
        pass

    def _rbf_kernel(self, x, h=-1):
        """
        Computes the Radial Basis Function (RBF) kernel and its gradient.
        
        Args:
            x (np.ndarray): An array of particles, shape (n_particles, n_dims).
            h (float): The bandwidth of the RBF kernel. If h=-1, a median heuristic is used.
            
        Returns:
            k (np.ndarray): The kernel matrix, shape (n_particles, n_particles).
            grad_k (np.ndarray): The gradient of the kernel, shape (n_particles, n_particles, n_dims).
        """
        sq_dist = np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) - 2 * np.dot(x, x.T)
        
        if h < 0:  # Median heuristic for bandwidth
            h = np.sqrt(0.5 * np.median(sq_dist) / np.log(x.shape[0] + 1))
        
        k = np.exp(-sq_dist / h**2 / 2)
        
        # Gradient of the kernel
        grad_k = -np.einsum('ij,ik->ijk', k, (x[:, np.newaxis, :] - x)) / h**2
        
        return k, grad_k

    def step(self, x, log_prob_grad, n_iter=1000, step_size=1e-3, alpha=0.9, adagrad=True):
        """
        Performs the SVGD update steps.
        
        Args:
            x (np.ndarray): Initial particles, shape (n_particles, n_dims).
            log_prob_grad (function): A function that takes particles `x` and returns the
                                      gradient of the log probability of the target distribution.
            n_iter (int): Number of update iterations.
            step_size (float): The learning rate or step size.
            alpha (float): Momentum parameter for Adagrad.
            adagrad (bool): Whether to use Adagrad for adaptive step sizes.
            
        Returns:
            x (np.ndarray): The final updated particles.
        """
        n_particles, n_dims = x.shape
        
        if adagrad:
            historical_grad = np.zeros_like(x)

        for i in tqdm(range(n_iter), desc="SVGD Steps"):
            # Calculate gradient of log-probability for all particles
            log_p_grad = log_prob_grad(x)
            
            # Calculate kernel and its gradient
            k, grad_k = self._rbf_kernel(x)
            
            # Calculate the SVGD update direction (phi)
            # This is the core of the SVGD algorithm
            phi = (np.dot(k, log_p_grad) + np.sum(grad_k, axis=0)) / n_particles
            
            # Update particles
            if adagrad:
                if i == 0:
                    historical_grad = phi**2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (phi**2)
                
                adj_grad = phi / (1e-6 + np.sqrt(historical_grad))
                x += step_size * adj_grad
            else:
                x += step_size * phi
                
        return x

def example_gmm():
    """
    An example of using SVGD to fit a 1D Gaussian Mixture Model (GMM).
    """
    # 1. Define the target distribution (a GMM)
    def log_prob(x):
        # Mixture of two Gaussians: N(-3, 1) and N(3, 1)
        p1 = np.exp(-0.5 * (x + 3)**2) / np.sqrt(2 * np.pi)
        p2 = np.exp(-0.5 * (x - 3)**2) / np.sqrt(2 * np.pi)
        return np.log(0.5 * p1 + 0.5 * p2)

    def log_prob_grad(x):
        # Gradient of the log probability of the GMM
        p1_unnorm = np.exp(-0.5 * (x + 3)**2)
        p2_unnorm = np.exp(-0.5 * (x - 3)**2)
        
        num = -p1_unnorm * (x + 3) - p2_unnorm * (x - 3)
        den = p1_unnorm + p2_unnorm
        
        return num / den

    # 2. Initialize particles
    n_particles = 100
    # Start with particles from a standard normal distribution
    initial_particles = np.random.normal(0, 1, (n_particles, 1))

    # 3. Run SVGD
    svgd = SVGD()
    final_particles = svgd.step(
        initial_particles.copy(), 
        log_prob_grad, 
        n_iter=2000, 
        step_size=0.1
    )

    # 4. Visualize the results
    plt.figure(figsize=(10, 6))
    
    # Plot initial distribution
    plt.hist(initial_particles, bins=30, density=True, alpha=0.6, label='Initial Particles (N(0,1))')
    
    # Plot final distribution
    plt.hist(final_particles, bins=30, density=True, alpha=0.8, label='Final Particles (SVGD)')
    
    # Plot true density
    x_range = np.linspace(-8, 8, 1000).reshape(-1, 1)
    true_density = np.exp(log_prob(x_range))
    plt.plot(x_range, true_density, 'r-', lw=2, label='True GMM Density')
    
    plt.title('SVGD Fitting a Gaussian Mixture Model')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    example_gmm()
