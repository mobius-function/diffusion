import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GaussianMixtureEM:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def generate_data(self, n_samples=1000, means=None, stds=None, weights=None):
        """Generate synthetic data from a mixture of normal distributions"""
        if means is None:
            means = [0, 5]
        if stds is None:
            stds = [1, 1.5]
        if weights is None:
            weights = [0.6, 0.4]

        # Update n_components based on provided parameters
        self.n_components = len(means)

        # Validate parameter consistency
        if len(stds) != self.n_components or len(weights) != self.n_components:
            raise ValueError(f"All parameters must have same length. Got means: {len(means)}, stds: {len(stds)}, weights: {len(weights)}")

        # Validate weights sum to 1
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        self.true_means = means
        self.true_stds = stds
        self.true_weights = weights

        # Generate samples
        data = []
        labels = []

        for i in range(n_samples):
            # Choose component based on weights
            component = np.random.choice(len(weights), p=weights)
            sample = np.random.normal(means[component], stds[component])
            data.append(sample)
            labels.append(component)

        return np.array(data), np.array(labels)

    def initialize_parameters(self, X):
        """Initialize parameters for EM algorithm"""
        n_samples = len(X)

        # Initialize means randomly from data range
        self.means = np.random.uniform(X.min(), X.max(), self.n_components)

        # Initialize standard deviations
        self.stds = np.ones(self.n_components)

        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components

        # Initialize responsibilities
        self.responsibilities = np.zeros((n_samples, self.n_components))

    def e_step(self, X):
        """Expectation step: compute responsibilities"""
        n_samples = len(X)

        for k in range(self.n_components):
            # Compute likelihood for each component
            likelihood = norm.pdf(X, self.means[k], self.stds[k])
            self.responsibilities[:, k] = self.weights[k] * likelihood

        # Normalize responsibilities
        total_responsibility = np.sum(self.responsibilities, axis=1, keepdims=True)
        self.responsibilities = self.responsibilities / total_responsibility

    def m_step(self, X):
        """Maximization step: update parameters"""
        n_samples = len(X)

        for k in range(self.n_components):
            # Effective number of points assigned to component k
            N_k = np.sum(self.responsibilities[:, k])

            # Update mean
            self.means[k] = np.sum(self.responsibilities[:, k] * X) / N_k

            # Update standard deviation
            self.stds[k] = np.sqrt(np.sum(self.responsibilities[:, k] * (X - self.means[k])**2) / N_k)

            # Update weight
            self.weights[k] = N_k / n_samples

    def compute_log_likelihood(self, X):
        """Compute log-likelihood of the data"""
        log_likelihood = 0
        for i, x in enumerate(X):
            likelihood = 0
            for k in range(self.n_components):
                likelihood += self.weights[k] * norm.pdf(x, self.means[k], self.stds[k])
            log_likelihood += np.log(likelihood)
        return log_likelihood

    def fit(self, X):
        """Fit the Gaussian mixture model using EM algorithm"""
        self.initialize_parameters(X)

        log_likelihood_history = []

        for iteration in range(self.max_iter):
            # E-step
            self.e_step(X)

            # M-step
            self.m_step(X)

            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(X)
            log_likelihood_history.append(log_likelihood)

            # Check for convergence
            if iteration > 0:
                if abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break

        self.log_likelihood_history = log_likelihood_history
        return self

    def predict(self, X):
        """Predict component assignments for new data"""
        self.e_step(X)
        return np.argmax(self.responsibilities, axis=1)

    def plot_results(self, X, true_labels=None):
        """Plot the data and fitted components"""
        plt.figure(figsize=(12, 8))

        # Plot 1: Data and true components (if available)
        plt.subplot(2, 2, 1)
        if true_labels is not None:
            for i in range(self.n_components):
                mask = true_labels == i
                plt.scatter(X[mask], np.zeros_like(X[mask]) + i*0.1, alpha=0.6, label=f'True Component {i}')
        else:
            plt.scatter(X, np.zeros_like(X), alpha=0.6, label='Data')
        plt.title('Original Data')
        plt.legend()

        # Plot 2: Fitted components
        plt.subplot(2, 2, 2)
        predicted_labels = self.predict(X)
        for i in range(self.n_components):
            mask = predicted_labels == i
            plt.scatter(X[mask], np.zeros_like(X[mask]) + i*0.1, alpha=0.6, label=f'Fitted Component {i}')
        plt.title('EM Algorithm Results')
        plt.legend()

        # Plot 3: Probability densities
        plt.subplot(2, 2, 3)
        x_range = np.linspace(X.min() - 2, X.max() + 2, 1000)

        # Plot true densities if available
        if hasattr(self, 'true_means'):
            total_true_density = np.zeros_like(x_range)
            for i in range(len(self.true_means)):
                density = self.true_weights[i] * norm.pdf(x_range, self.true_means[i], self.true_stds[i])
                plt.plot(x_range, density, '--', alpha=0.7, label=f'True Component {i}')
                total_true_density += density
            plt.plot(x_range, total_true_density, 'k--', linewidth=2, label='True Mixture')

        # Plot fitted densities
        total_fitted_density = np.zeros_like(x_range)
        for i in range(self.n_components):
            density = self.weights[i] * norm.pdf(x_range, self.means[i], self.stds[i])
            plt.plot(x_range, density, '-', alpha=0.7, label=f'Fitted Component {i}')
            total_fitted_density += density
        plt.plot(x_range, total_fitted_density, 'r-', linewidth=2, label='Fitted Mixture')

        plt.hist(X, bins=30, density=True, alpha=0.3, color='gray')
        plt.title('Probability Densities')
        plt.legend()

        # Plot 4: Log-likelihood convergence
        plt.subplot(2, 2, 4)
        plt.plot(self.log_likelihood_history)
        plt.title('Log-likelihood Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create EM model
    em_model = GaussianMixtureEM()

    # Generate synthetic data
    data, true_labels = em_model.generate_data(n_samples=1000, means=[2, 8], stds=[1, 1.5], weights=[0.3, 0.7])

    print("True parameters:")
    print(f"Means: {em_model.true_means}")
    print(f"Standard deviations: {em_model.true_stds}")
    print(f"Weights: {em_model.true_weights}")

    # Fit the model
    em_model.fit(data)

    print("\nEstimated parameters:")
    print(f"Means: {em_model.means}")
    print(f"Standard deviations: {em_model.stds}")
    print(f"Weights: {em_model.weights}")

    # Plot results
    em_model.plot_results(data, true_labels)