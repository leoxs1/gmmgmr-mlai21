import numpy as np
from scipy.stats import multivariate_normal
import scipy
# from utilities import *
import csv


def save_model(gmrobj, filename):
    """
    Save fitted model means and covariances to a csv file

    :param gmrobj: GMMGMR object
    :param filename: filename to save to
    :return: None
    """
    cov_filename = filename + '_cov.csv'
    mean_filename = filename + '_means.csv'
    model_cov = gmrobj.covariances.tolist()
    model_mean = gmrobj.means.tolist()

    with open(cov_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(model_cov)
    with open(mean_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(model_mean)


def load_model(filename):
    """
    Load a saved model

    :param filename: string
    :return: GMMGMR object with model means and covariances loaded in
    """
    cov_filename = filename + '_cov.csv'
    mean_filename = filename + '_means.csv'

    with open(cov_filename, 'r') as f:
        reader = csv.reader(f)
        model_cov_read = list(reader)

    model_cov_shaped = []
    for row in model_cov_read:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        model_cov_shaped.append(nwrow)
    model_cov = np.array(model_cov_shaped)

    with open(mean_filename, 'r') as f:
        reader = csv.reader(f)
        model_mean_read = list(reader)

    model_mean_shaped = []
    for row in model_mean_read:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        model_mean_shaped.append(nwrow)
    model_mean = np.array(model_mean_shaped)

    gmrobj = GMMGMR(model_mean.shape[0])
    gmrobj.means = model_mean
    gmrobj.covariances = model_cov
    return gmrobj


class GMMGMR:
    def __init__(self, k):
        """
        This class implements a Gaussian Mixture Model using the EM algorithm

        :param k: (int) number of gaussians to cluster the data to
        """
        self.k = k
        self.max_iterations = 200

        self.means = None
        self.covariances = None
        self.weights = None
        self.phi = None

        self.means_inputs = None
        self.means_outputs = None
        self.cov_inputs = None
        self.cov_outputs = None
        self.cov_inout = None
        self.cov_outin = None

    def fit(self, features):
        """
        Fit GMM to inputs using k number of gaussians

        :param features: (np.array(N, dim)) N samples with dimension dim
        """
        # Use KMeans to initialize GMM means
        kmeans = KMeans(self.k)
        kmeans.fit(features)
        self.means = kmeans.means

        # Initialize the covariance matrix
        # print(features.shape[1])
        self.covariances = np.empty((self.k, features.shape[1], features.shape[1]))
        self.covariances = [np.eye(features.shape[1]) for _ in range(self.k)]

        # Initialize the weights
        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.zeros((features.shape[0], self.k))
        self.weights.fill(1/self.k)

        # Compute log_likelihood under initial random covariance and KMeans means.
        prev_log_likelihood = -float('inf')
        log_likelihood = self._total_log_likelihood(features)

        # While the log_likelihood is increasing significantly, or max_iterations has
        # not been reached, continue EM until convergence.
        n_iter = 0
        print(">\tstarting EM")
        while abs(log_likelihood - prev_log_likelihood) > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            # update the weights and phi (mean of the weights)
            self.weights, self.phi = self._e_step(features)
            self._m_step(features)
            log_likelihood = self._total_log_likelihood(features)
            n_iter += 1

    def predict(self, queries, input_dim, output_dim):
        """
        Gaussian Mixture Regression to predict outputs

        :param queries: inputs to predict
        :param input_dim: int of the number of input dimensions
        :param output_dim: int of the number of output dimensions
        :return: predictions and covariance of prediction (uncertainty)
        """
        self.means_inputs = self.means[:, 0:input_dim]
        self.means_outputs = self.means[:, input_dim:]
        self.cov_inputs = self.covariances[:, 0:input_dim, 0:input_dim]
        self.cov_outputs = self.covariances[:, input_dim:, input_dim:]
        self.cov_inout = self.covariances[:, 0:input_dim, input_dim:]
        self.cov_outin = self.covariances[:, input_dim:, 0:input_dim]

        predictions = np.zeros((queries.shape[0], output_dim))
        beta = None
        for i in range(queries.shape[0]):
            beta = self._calculate_beta(queries[i])
            unweighted_pred = self._calculate_kernel_predictions(queries[i], output_dim)

            if output_dim == 1:
                predictions[i] = np.sum(np.multiply(np.reshape(beta, (self.k, 1)), unweighted_pred), axis=0)
            else:
                predict_temp = np.zeros((1, output_dim))
                for k in range(self.k):
                    predict_temp += beta[k] * unweighted_pred[k, :]
                predictions[i] = predict_temp

        pred_cov_arr = np.zeros((self.k, output_dim, output_dim))
        for k in range(self.k):
            pred_cov_temp = self.cov_outputs[k] - self.cov_outin[k] @ np.linalg.pinv(self.cov_inputs[k]) @ self.cov_inout[k]
            pred_cov_arr[k, :, :] = (beta[k] ** 2) * pred_cov_temp
        pred_cov = np.sum(pred_cov_arr, axis=0)

        return predictions, pred_cov

    def bic(self, targets, predictions):
        """
        Calculate Bayesian Information Criterion (BIC) of a given model
        Code adapted from https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/bic.py

        :param targets: np.array(N x output_dim) ground truth
        :param predictions: np.array(N x output_dim) predictions
        :return: BIC value (the lower the better)
        """
        num_samples = targets.shape[0]
        res = predictions - targets
        rss = np.sum(np.power(res, 2))
        BIC = num_samples * np.log(rss / num_samples) + self.k * np.log(num_samples)
        return BIC

    def _e_step(self, features):
        """
        Expectation step in EM algorithm.
        Update the weights (responsibility of each gaussian for each datapoint)
        and phi (overall responsibility of each gaussian for the whole dataset)

        :param features: np.array(N, dim) inputs
        :return: None, updates class attributes self.weights and self.phi
        """
        # calculate for each gaussian
        weights = np.zeros((features.shape[0], self.k))
        for i in range(self.k):
            weight = self._weight_update(features, i).T
            weights[:, i] = weight[:, 0]
        phi = np.mean(weights, axis=0)
        return weights, phi

    def _m_step(self, features):
        """
        Maximization step in EM algorithm.
        Update the means and covariances of the gaussians given the weight and phi
        calculated from the E step

        :param features: np.array(N, dim) inputs
        :return: None, updates class attributes self.means and self.covariances
        """
        means = np.empty((self.k, features.shape[1]))
        covariances = np.empty((self.k, features.shape[1], features.shape[1]))

        total_weight = np.sum(self.weights, axis=0)     # flatten to row
        num_samples = features.shape[0]

        for j in range(self.k):
            mean_numer_arr = np.empty((num_samples, features.shape[1]))
            cov_numer_arr = np.empty((num_samples, features.shape[1], features.shape[1]))

            for i in range(num_samples):
                mean_numer_arr[i, :] = np.multiply(self.weights[i, j], features[i, :])
            mean_numer = np.sum(mean_numer_arr, axis=0)   # get a row vector for mean of each feature
            mean = mean_numer / total_weight[j]

            for i in range(num_samples):
                for row in range(features.shape[1]):
                    for col in range(features.shape[1]):
                        sq = (features[i, row] - mean[row]) * (features[i, col] - mean[col])
                        cov_numer_arr[i, row, col] = np.multiply(self.weights[i, j], sq)

            cov_numer = np.sum(cov_numer_arr, axis=0)   # get a row vector for mean of each feature
            cov = cov_numer / total_weight[j]

            means[j, :] = mean
            covariances[j, :, :] = cov
        self.means = means
        self.covariances = covariances

    def _make_good(self, matrix):
        """
        Error catching for singular and non-PSD covariance matrix

        :param matrix: covariance matrix to catch
        :return: hopefully a non-singular and PSD covariance matrix
        """
        det = np.linalg.det(matrix)
        if det < 1e-8:
            # create a random symmetric noise matrix to perturb by
            noise_mat = np.eye(matrix.shape[0]) * np.random.uniform(low=0, high=0.5)
            matrix = matrix + noise_mat
        # Check that all eigenvalues are positive
        eig = np.linalg.eigvals(matrix)

        flag = np.all(eig > 0)
        scale = 3

        if flag:
            return matrix
        else:
            matrix_shift = matrix + np.identity(matrix.shape[0]) * (-eig.min() * scale + np.spacing(eig.min()))
            return matrix_shift

    def _log_likelihood(self, features, k_idx):
        """
        Compute the log likelihood that each input datapoint belongs to a given gaussian

        :param features: (np.array(N, dim)) inputs
        :param k_idx: index of gaussian to check
        :return: log-likelihood of each input (np.array(N))
        """
        self.covariances[k_idx] = self._make_good(self.covariances[k_idx])
        log_prob = multivariate_normal.logpdf(features, self.means[k_idx], self.covariances[k_idx])
        log_likelihood = log_prob + np.log(self.phi[k_idx])
        return log_likelihood

    def _total_log_likelihood(self, features):
        """
        Calculate total log likelihood, for checking convergence
        :param features: training samples
        :return: the total log likelihood
        """
        denom = [self._log_likelihood(features, j) for j in range(self.k)]
        return np.sum(denom)

    def _weight_update(self, features, k):
        """
        Compute responsibility of given gaussian for each input datapoint

        :param features: (np.array(N, dim)) the input data
        :param k: the gaussian to compute for
        :return: weights for this k (np.array(N))
        """
        numerator = self._log_likelihood(features, k)
        denom = np.array([self._log_likelihood(features, j) for j in range(self.k)])

        # logsumexp
        max_denom = denom.max(axis=0, keepdims=True)
        denom_sum = max_denom + np.log(np.sum(np.exp(denom - max_denom), axis=0))
        weights = np.exp(numerator - denom_sum)
        return weights

    def _calculate_beta(self, query):
        """
        Calculate beta in the GMR algorithm

        :param query: query point (input)
        :return: beta array
        """
        beta = np.zeros(self.k)
        total_prob_k = 0
        for k in range(self.k):
            self.cov_inputs[k] = self._make_good(self.cov_inputs[k])
            beta[k] = multivariate_normal.pdf(query, self.means_inputs[k], self.cov_inputs[k])
            total_prob_k += beta[k]
        beta /= total_prob_k
        return beta

    def _calculate_kernel_predictions(self, query, output_dim):
        """
        Calculate prediction of each kernel

        :param query:
        :param output_dim:
        :return:
        """
        y_preds = np.zeros((self.k, output_dim))
        for k in range(self.k):
            y_pred = self.means_outputs[k] + self.cov_outin[k] @ np.linalg.pinv(self.cov_inputs[k]) \
                     @ (query - self.means_inputs[k])
            y_preds[k, :] = y_pred
        return y_preds


class KMeans:
    def __init__(self, k):
        """
        This class implements the KMeans to initialize the GMM

        :param k: number of clusters
        """
        self.k = k
        self.means = None
        self.assignments = None

    def update_assignments(self, features):
        """
        Update the assignments of each sample
        :param features: (np.array(N, dim)) inputs
        :return: assigments (np.array(N)), assignments of each input
        """
        # for all features, calculate all the euclidean distances from all means
        distances = np.empty([features.shape[0], self.k])
        for i in range(features.shape[0]):
            for j in range(self.k):
                distances[i, j] = np.linalg.norm(features[i, :]-self.means[j, :])

        # choose the closest mean and assign every sample to only one mean
        assignments = np.argmin(distances, axis=1)
        return assignments

    def update_means(self, features):
        """
        Function to update centroid of clusters

        :param features: np.array(N x dim) of inputs
        :return:
        """
        for i in range(self.k):
            points_idx = np.where(self.assignments == i)
            points = features[points_idx, :][0]
            mean = np.mean(points, axis=0)
            self.means[i, :] = mean

    def fit(self, features):
        """
        Fit KMeans using k clusters.

        :param features: np.array(N x dim) of inputs
        """
        # randomly initialize the means
        num_features = features.shape[0]
        means_idx = np.random.choice(np.arange(num_features), size=self.k, replace=False)
        self.means = features[means_idx, :]
        repeat = True

        # do update_assignments and update_means until no more changes in assignments
        while repeat:
            assignments = self.update_assignments(features)
            if np.array_equal(assignments, self.assignments):
                repeat = False
            else:
                self.assignments = assignments
                self.update_means(features)
