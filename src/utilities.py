"""
Module for utility functions

Arrow plot fpr trajectory
Helper function for error computation (question 6)
Augmenting controls for time synchronization
Compute error on the filtered trajectory
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections


class Standardize:
    def __init__(self):
        """
        This class implements a standardization scaler for data
        Use this class for each dimension of the data

        Code adaoted from:
        https://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html
        """
        self.mean = None
        self.scale = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.scale = np.std(data - self.mean, axis=0)
        return self

    def transform(self, data):
        return (data - self.mean) / self.scale

    def transform_back(self, data):
        return (data * self.scale) + self.mean


def normalize_angle(rad):
    """
    Normalise an angle in radians to range -pi to pi

    :param rad: angle in radians
    :return: normalized angle in radians
    """
    angle = rad % (2 * np.pi)                   # reduce angle
    angle = (angle + 2 * np.pi) % (2 * np.pi)   # make angle positive
    if angle > np.pi:
        angle = angle - (2 * np.pi)
    return angle


def calculate_error_metrics(targets, prediction):
    """
    Error metrics MSE, MAE, R^2
    :param targets:
    :param prediction:
    :return:
    """
    error = targets - prediction
    sumsqerr = np.sum(np.power(error, 2), axis=0)
    mse = sumsqerr / error.shape[0]

    abserr = np.abs(error)
    sumabserr = np.sum(abserr, axis=0)
    mae = sumabserr / error.shape[0]

    error_norm = error / targets
    abserr_norm = np.abs(error_norm)
    sumabserr_norm = np.sum(abserr_norm, axis=0)
    mape = sumabserr_norm / error.shape[0]

    meantargets = np.mean(targets, axis=0)
    denom = np.sum(np.power((targets-meantargets), 2), axis=0)
    r2score = 1 - (sumsqerr / denom)

    return mse, mae, mape, r2score


def create_test_datasets():
    """
    Create the toy datasets
    :return:
    """
    # create some test datasets
    train = collections.OrderedDict()
    test = collections.OrderedDict()

    x = np.arange(0, 5, 0.01)
    np.random.shuffle(x)    # shuffle the points
    y = x + 3 + np.random.normal(0, 0.2, x.shape[0])    # add some noise
    num_test = int(0.2 * x.shape[0])
    num_train = x.shape[0] - num_test
    trainx = x[:num_train]
    testx = x[num_train:]
    trainy = y[:num_train]
    testy = y[num_train:]
    # create training set (all input and output dimensions to cluster
    train_set = np.concatenate((np.reshape(trainx, (num_train, 1)), np.reshape(trainy, (num_train, 1))), axis=1)

    train['pos_linear'] = {}
    train['pos_linear']['train_set'] = train_set
    train['pos_linear']['trainx'] = trainx
    train['pos_linear']['trainy'] = trainy
    test['pos_linear'] = {}
    test['pos_linear']['testx'] = testx
    test['pos_linear']['testy'] = testy

    x = np.arange(0, 5, 0.01)
    np.random.shuffle(x)    # shuffle the points
    y = -x - 5 + np.random.normal(0, 0.2, x.shape[0])    # add some noise
    num_test = int(0.2 * x.shape[0])
    num_train = x.shape[0] - num_test
    trainx = x[:num_train]
    testx = x[num_train:]
    trainy = y[:num_train]
    testy = y[num_train:]
    # create training set (all input and output dimensions to cluster
    train_set = np.concatenate((np.reshape(trainx, (num_train, 1)), np.reshape(trainy, (num_train, 1))), axis=1)

    train['neg_linear'] = {}
    train['neg_linear']['train_set'] = train_set
    train['neg_linear']['trainx'] = trainx
    train['neg_linear']['trainy'] = trainy
    test['neg_linear'] = {}
    test['neg_linear']['testx'] = testx
    test['neg_linear']['testy'] = testy

    x = np.arange(0, 2*np.pi, 0.01)
    np.random.shuffle(x)    # shuffle the points
    y = np.sin(x) + np.random.normal(0, 0.2, x.shape[0])    # add some noise
    num_test = int(0.2 * x.shape[0])
    num_train = x.shape[0] - num_test
    trainx = x[:num_train]
    testx = x[num_train:]
    trainy = y[:num_train]
    testy = y[num_train:]
    # create training set (all input and output dimensions to cluster
    train_set = np.concatenate((np.reshape(trainx, (num_train, 1)), np.reshape(trainy, (num_train, 1))), axis=1)

    train['sine'] = {}
    train['sine']['train_set'] = train_set
    train['sine']['trainx'] = trainx
    train['sine']['trainy'] = trainy
    test['sine'] = {}
    test['sine']['testx'] = testx
    test['sine']['testy'] = testy

    return train, test


def kfolds_cross_validation(data, k=5):
    """
    splitter for k-folds cross validation
    :param data: data to split
    :param k: number of folds
    :return: split data as a 3d array, first dim is the number of folds
    """
    num_samples = int(data.shape[0]/k)
    # print(num_samples)
    data_folds = np.zeros((k, num_samples, data.shape[1]))
    for i in range(k):
        samples = data[i::k, :]
        samples = samples[:num_samples, :]
        data_folds[i, :, :] = samples
    return data_folds


def plot_confidence_ellipse(means, cov, axs, color, scale=2.0):
    """
    Code for plotting the ellipses adapted from this example from scikit-learn
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html

    :param means: means of gaussians
    :param cov: cov of gaussians
    :param axs: axis to plot on
    :return:
    """
    for i in range(means.shape[0]):
        eigvals, eigvecs = np.linalg.eigh(cov[i, :])
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.arctan2(*eigvecs[:, 0][::-1])
        width, height = scale * np.sqrt(eigvals)

        ell = matplotlib.patches.Ellipse(xy=means[i], width=2.0*width, height=2.0*height,
                                         angle=np.degrees(angle), color=color, zorder=0)
        ell.set_clip_box(axs.bbox)
        ell.set_alpha(0.3)
        ell.set_label = 'GMM clusters'
        axs.add_patch(ell)

    axs.scatter(means[:, 0], means[:, 1], color='k', s=20, label='GMM means', zorder=100)


def make_arrow_plot(trajectory, label='trajectory', line=False, axis=None, ms=15, interval=50, arrowlength=1,
                    scale=5, color='tab:blue'):
    """
    Create the trajectory plot with arrows

    :param trajectory: The trajectory to plot
    :param label: The label for the plot
    :param line: To plot the connecting line or not
    :param axis: The axis to plot on
    :param ms: scatter marker size
    :param interval: The interval between data points to plot arrows
    :param arrowlength: The length of the arrows
    :param scale: The scaling factor for the arrows
    :param color: The colour of the arrows
    """
    # subsample the trajectory for arrow origins and directions
    arrow_x = trajectory[0::interval, 0]
    arrow_y = trajectory[0::interval, 1]
    arrow_heading = trajectory[0::interval, 2]

    # Calculate the x and y components of the arrow vector
    arrow_vec_x = arrowlength * np.cos(arrow_heading)
    arrow_vec_y = arrowlength * np.sin(arrow_heading)

    if axis is None:
        if line is True:
            plt.plot(trajectory[:, 0], trajectory[:, 1], color='k', zorder=0)
        plt.scatter(arrow_x, arrow_y, s=ms, marker="o", color=color, label=label)
        plt.quiver(arrow_x, arrow_y, arrow_vec_x, arrow_vec_y, color=color, angles='xy',
                   scale_units='height', scale=scale, headwidth=1, headlength=0)
    else:
        if line is True:
            axis.plot(trajectory[:, 0], trajectory[:, 1], color='k', zorder=0)
        axis.scatter(arrow_x, arrow_y, s=ms, marker="o", color=color, label=label)
        axis.quiver(arrow_x, arrow_y, arrow_vec_x, arrow_vec_y, color=color, angles='xy',
                    scale_units='height', scale=scale, headwidth=1, headlength=0)


def compute_globalxy_error(groundtruth, signature, xy):
    """
    Helper function for error computation

    :param groundtruth: Ground truth landmark location in world coordinates
    :param signature: The landmark signature
    :param xy: The predicted landmark position in the world coordinates
    :return: The error between the prediction and the ground truth
    """
    idx = np.where(groundtruth[:, 0] == signature)
    groundtruth_xy = groundtruth[idx, 1:3]
    return np.squeeze(xy - groundtruth_xy)


def augment_controls(controls, measurements):
    """
    Split up the controls into smaller timesteps to encompass the measurement timesteps

    :param controls: The control data from _Odometry.dat
    :param measurements: The measurements data from _Measurements.dat
    :return: The new controls array with the durations in the last column
    """
    timesteps_arr = np.sort(np.concatenate((measurements[:, 0], controls[:, 0]), axis=0))
    control_time = np.unique(timesteps_arr)
    prev_control = np.zeros(2) # forward and angular velocity control
    new_controls = np.array([[control_time[0], controls[0, 1], controls[0, 2]]])

    for t in control_time[1:]:
        if t in controls[:, 0]:
            idx = np.squeeze(np.where(controls[:, 0] == t))
            prev_control = np.squeeze(controls[idx, 1:3])
            add_control = np.array([[t, prev_control[0], prev_control[1]]])
            new_controls = np.concatenate((new_controls, add_control), axis=0)
        else:
            add_control = np.array([[t, prev_control[0], prev_control[1]]])
            new_controls = np.concatenate((new_controls, add_control), axis=0)

    duration_arr = np.zeros((new_controls.shape[0], 1))

    # Check if the last control command is before or after the last measurement
    if control_time[-1] > measurements[-1, 0]:
        duration_arr[-1] = 1 / 67
    else:
        duration_arr[-1] = 0.0
    for i in range(control_time.shape[0] - 1):
        duration_arr[i] = control_time[i+1] - control_time[i]
    output_controls = np.concatenate((new_controls, duration_arr), axis=1)

    return output_controls


def compute_trajectory_error(trajectory, groundtruth):
    """
    Compute the error for x, y, and theta
    Compute MSE for each of x, y, and theta

    :param trajectory: The filtered trajectory
    :param groundtruth: The ground truth trajectory to compare against
    :return: The error at shared timesteps and the MSE over the trajectory
    """
    timesteps = []
    for t in trajectory[:, 0]:
        if t in groundtruth[:, 0]:
            timesteps.append(t)
    timesteps = np.array(timesteps)
    traj = np.zeros((timesteps.shape[0], 4))
    traj[:, 0] = timesteps
    gt_traj = np.zeros((timesteps.shape[0], 4))
    gt_traj[:, 0] = timesteps

    # Extract relevant trajectory points for comparison
    for i in range(timesteps.shape[0]):
        traj_idx = np.squeeze(np.where(trajectory[:, 0] == timesteps[i]))
        if traj_idx.size > 1:
            traj_idx = traj_idx[-1]

        traj[i, :] = trajectory[traj_idx, :]
        gt_idx = np.squeeze(np.where(groundtruth[:, 0] == timesteps[i]))

        traj[i, :] = trajectory[traj_idx, :]
        gt_traj[i, 1:] = groundtruth[gt_idx, 1:]

    error = gt_traj - traj
    error[:, 0] = timesteps

    # Calculate MSE for 3 dimensions
    error_no_time = error[:, 1:]
    # RMSE = np.sqrt(np.sum(np.power(error_no_time, 2), axis=0) / timesteps.shape[0])
    MSE = np.sum(np.power(error_no_time, 2), axis=0) / timesteps.shape[0]

    return error, MSE
