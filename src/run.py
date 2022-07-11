import collections

from gmmgmr import GMMGMR, load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from utilities import *
from dataloader import load_kfolds_data
from IKinModel import *
import time


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
color = ['tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan']
np.random.seed(0)


def justify_learning_aim(dataset):
    """
    Part 1 with naive ik model
    :param dataset:
    :return:
    """
    current_pose_arr = dataset[:, 1:4]
    next_pose_arr = dataset[:, 4:7]
    dt_arr = dataset[:, 7:8]
    gt_control = dataset[:, 8:] # [v, w]
    modelled_control = np.zeros((dataset.shape[0], 2))
    for i in range(dataset.shape[0]):
        control_temp = inverse_kinematics(current_pose_arr[i, :], next_pose_arr[i, :], dt_arr[i, :])
        modelled_control[i, :] = control_temp

    # plot on vw plane to show spread
    plt.rcParams["figure.figsize"] = (7, 4)
    plt.scatter(modelled_control[:, 0], modelled_control[:, 1], s=1, alpha=0.5, label='modelled controls', color=color[1])
    plt.scatter(gt_control[:, 0], gt_control[:, 1], s=1, label='ground truth controls', color=color[0])
    plt.ylim([-1, 1])
    plt.xlim([-0.1, 0.15])
    plt.xlabel('v (m/s)')
    plt.ylabel(r'$\omega$ (rad/s)')
    plt.legend()
    plt.title("Ground truth and modelled control commands", fontsize=15)
    plt.tight_layout()
    plt.savefig('figures/IKmodelcompare.png')
    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def justify_learning_aim_with_acclim(dataset):
    """
    Part 1 with ik model that has acceleration limits applied
    :param dataset:
    :return:
    """
    np.seterr(all='ignore')
    current_pose_arr = dataset[:, 1:4]
    next_pose_arr = dataset[:, 4:7]
    prev_commands_arr = dataset[:, 7:9]
    dt_arr = dataset[:, 9:10]
    gt_control = dataset[:, 10:] # [v, w]
    modelled_control = np.zeros((dataset.shape[0], 2))
    for i in range(dataset.shape[0]):
        control_temp = inverse_kinematics_acclim(current_pose_arr[i, :], next_pose_arr[i, :], prev_commands_arr[i, :], dt_arr[i, :])
        modelled_control[i, :] = control_temp
    np.seterr(all=None)

    # plot on vw plane to show spread
    plt.rcParams["figure.figsize"] = (7, 4)
    plt.scatter(modelled_control[:, 0], modelled_control[:, 1], s=1, alpha=0.5, label='modelled controls', color=color[1])
    plt.scatter(gt_control[:, 0], gt_control[:, 1], s=1, label='ground truth controls', color=color[0])
    plt.ylim([-0.7, 0.7])
    plt.xlim([-0.03, 0.1])
    plt.xlabel('v (m/s)')
    plt.ylabel(r'$\omega$ (rad/s)')
    plt.legend()
    plt.title("Ground truth and modelled control commands \n with acceleration limits", fontsize=15)
    plt.tight_layout()
    plt.savefig('figures/IKmodelcompare_big.png')
    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def test_simple_functions():
    """
    Part 2: Testing on simple sine function, 1 input 1 output
    """
    print("Demonstrate algorithm working on simple test dataset")
    train_dict, test_dict = create_test_datasets()
    k_arr = [1, 1, 3]
    func_keys = ['pos_linear', 'neg_linear', 'sine']
    title_arr = ['linear with positive slope', 'linear with negative slope', 'sine function']

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for i in range(3):
        print("Test function", i)
        print("> Begin fitting GMM, this will take some time, thanks for your patience!")
        func = func_keys[i]
        gmr = GMMGMR(k_arr[i])
        gmr.fit(train_dict[func]['train_set'])
        print(">\tdone fitting")
        pred, pred_cov = gmr.predict(test_dict[func]['testx'], 1, 1)
        print(">\tdone predicting")

        print("prediction covariance")
        print(pred_cov)

        plot_confidence_ellipse(gmr.means, gmr.covariances, axs[i], color='cyan')
        axs[i].scatter(train_dict[func]['trainx'], train_dict[func]['trainy'], color='orange', label='training data', s=1)
        axs[i].scatter(test_dict[func]['testx'], pred, color='r', label='predictions', s=1)
        axs[i].scatter(test_dict[func]['testx'], test_dict[func]['testy'], color='b', label='test groundtruth', s=1)
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        axs[i].set_box_aspect(1)
        title = title_arr[i] + '\nconditional variance: ' + str("{0:.4f}".format(np.squeeze(pred_cov)))
        axs[i].set_title(title)

    fig.suptitle('Testing GMM-GMR on simple functions with known groundtruth', fontsize=15)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    plt.legend()
    plt.savefig('figures/toy_functions.png')
    plt.show()


def plot_bic_from_sklearn():
    """
    Plotting the BIC from parameter search using sklearn
    :return:
    """
    bic_array = np.loadtxt('bic_from_sklearn/sklearn_bic.csv', delimiter=",")
    k_list = np.arange(start=1, stop=bic_array.shape[0]+1, step=1)
    bic_grad = np.gradient(bic_array)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(k_list[:40], bic_array[:40])
    ax1.set_xlabel('k')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC with increasing k')
    # ax1.set_box_aspect(1)

    ax2.plot(k_list[:40], bic_grad[:40])
    ax2.set_xlabel('k')
    ax2.set_ylabel('grad(BIC)')
    ax2.set_title('grad(BIC) with increasing k')
    # ax2.set_box_aspect(1)

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle("Selection of k from BIC scores generated using sklearn \n"
                 "data with 9 dimensions (input + output)", fontsize=15)
    plt.savefig('figures/sklearn_bic.png')
    plt.show()


def plot_bic_from_sklearn_acclim():
    """
    Plotting the BIC from parameter search using sklearn
    for expanded dataset
    :return:
    """
    bic_array = np.loadtxt('bic_from_sklearn/sklearn_bic_big.csv', delimiter=",")
    k_list = np.arange(start=1, stop=bic_array.shape[0]+1, step=1)
    bic_grad = np.gradient(bic_array)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(k_list[:40], bic_array[:40])
    ax1.set_xlabel('k')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC with increasing k')
    # ax1.set_box_aspect(1)

    ax2.plot(k_list[:40], bic_grad[:40])
    ax2.set_xlabel('k')
    ax2.set_ylabel('grad(BIC)')
    ax2.set_title('grad(BIC) with increasing k')
    # ax2.set_box_aspect(1)

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle("Selection of k from BIC scores generated using sklearn \n"
                 "data with 11 dimensions (input + output)", fontsize=15)
    plt.savefig('figures/sklearn_bic_big.png')
    plt.show()


def plot_prediction_results_ellipse():
    """
    Prediction results of smaller dataset (plot)
    :return:
    """
    # load downsampled, shuffled dataset
    train = np.loadtxt("datasets/train_shuffle_ordered.csv", delimiter=",")
    test = np.loadtxt("datasets/test_ordered.csv", delimiter=",")
    num_train_samples = train.shape[0]
    num_test_samples = test.shape[0]
    dim = train.shape[1]
    # standardize data
    train_scaled = np.zeros((num_train_samples, dim))
    test_scaled = np.zeros((num_test_samples, dim))
    scaler_list = []
    for i in range(dim):
        scaler = Standardize().fit(train[:, i])
        scaler_list.append(scaler)
        train_scaled[:, i] = np.reshape(scaler.transform(train[:, i]), (num_train_samples, ))
        test_scaled[:, i] = np.reshape(scaler.transform(test[:, i]), (num_test_samples, ))

    # partition into input and output
    test_scaled_features = test_scaled[:, :7]
    test_scaled_targets = test_scaled[:, 7:]

    gmr7in2out = load_model('models/gmrds0_shuffle_model')

    print("Starting prediction!")
    start_time = time.time()
    pred_test, pred_test_cov = gmr7in2out.predict(test_scaled_features, 7, 2)
    print("prediction time: ", time.time()-start_time)
    mse_test, mae_test, _, r2_test = calculate_error_metrics(test_scaled_targets, pred_test)
    print("testing mse", mse_test)
    print("testing mae", mae_test)
    print("testing r2 score", r2_test)
    print("testing prediction covariance")
    print(pred_test_cov)

    # unscale the data
    unscale_pred_test_v = np.reshape(scaler_list[7].transform_back(pred_test[:, 0]), (num_test_samples, 1))
    unscale_pred_test_w = np.reshape(scaler_list[8].transform_back(pred_test[:, 1]), (num_test_samples, 1))

    # extract the ellipses to draw
    yycov = gmr7in2out.cov_outputs
    yymeans = gmr7in2out.means_outputs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    plot_confidence_ellipse(yymeans, yycov, ax1, color='cyan', scale=1.0)
    ax1.scatter(test_scaled[:, 7], test_scaled[:, 8], label='ground truth', color=color[0], s=2)
    ax1.scatter(pred_test[:, 0], pred_test[:, 1], label='predictions', color=color[1], s=2, alpha=0.7)
    ax1.set_xlabel('v standardized')
    ax1.set_ylabel(r'$\omega$ standardized')
    title1 = 'Standardized results with confidence ellipses'
    ax1.set_title(title1)
    ax1.set_box_aspect(1)
    ax1.legend()

    ax2.scatter(test[:, 7], test[:, 8], label='ground truth', color=color[0], s=2)
    ax2.scatter(unscale_pred_test_v, unscale_pred_test_w, label='predictions', color=color[1], s=2, alpha=0.7)
    ax2.set_xlabel('v (m/s)')
    ax2.set_ylabel(r'$\omega$ (rad/s)')
    title2 = 'Prediction on test data transformed \n back to not standardized scale'
    ax2.set_title(title2)
    ax2.set_box_aspect(1)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    title = 'Prediction on test data with 9 inputs and 2 outputs \n' \
            'v MSE: ' + str("{0:.4f}".format(mse_test[0])) + \
            r', $\omega$ MSE: ' + str("{0:.4f}".format(mse_test[1])) + '\n' + \
            r'v $R^2$ score: ' + str("{0:.4f}".format(r2_test[0])) + \
            r', $\omega$ $R^2$ score: ' + str("{0:.4f}".format(r2_test[1]))
    fig.suptitle(title, fontsize=15)
    plt.legend()
    plt.savefig('figures/test_predictions_ellipses.png')
    plt.show()


def plot_prediction_results_acclim():
    """
    Prediction results of bigger dataset (plot)
    :return:
    """
    # load downsampled, shuffled dataset
    train = np.loadtxt("datasets/train_shuffle_big.csv", delimiter=",")
    test = np.loadtxt("datasets/test_big.csv", delimiter=",")
    num_train_samples = train.shape[0]
    num_test_samples = test.shape[0]
    dim = train.shape[1]
    # standardize data
    train_scaled = np.zeros((num_train_samples, dim))
    test_scaled = np.zeros((num_test_samples, dim))
    scaler_list = []
    for i in range(dim):
        scaler = Standardize().fit(train[:, i])
        scaler_list.append(scaler)
        train_scaled[:, i] = np.reshape(scaler.transform(train[:, i]), (num_train_samples, ))
        test_scaled[:, i] = np.reshape(scaler.transform(test[:, i]), (num_test_samples, ))

    # partition into input and output
    test_scaled_features = test_scaled[:, :9]
    test_scaled_targets = test_scaled[:, 9:]

    gmr9in2out = load_model('models/gmrds0_big_shuffle_model')

    print("Starting prediction!")
    start_time = time.time()
    pred_test, pred_test_cov = gmr9in2out.predict(test_scaled_features, 9, 2)
    print("prediction time: ", time.time()-start_time)
    mse_test, mae_test, _, r2_test = calculate_error_metrics(test_scaled_targets, pred_test)
    print("testing mse", mse_test)
    print("testing mae", mae_test)
    print("testing prediction covariance")
    print(pred_test_cov)

    # unscale the data
    unscale_pred_test_v = np.reshape(scaler_list[9].transform_back(pred_test[:, 0]), (num_test_samples, 1))
    unscale_pred_test_w = np.reshape(scaler_list[10].transform_back(pred_test[:, 1]), (num_test_samples, 1))

    # extract the ellipses to draw
    yycov = gmr9in2out.cov_outputs
    yymeans = gmr9in2out.means_outputs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    plot_confidence_ellipse(yymeans, yycov, ax1, color='cyan', scale=1.0)
    ax1.scatter(test_scaled[:, 9], test_scaled[:, 10], label='ground truth', color=color[0], s=2)
    ax1.scatter(pred_test[:, 0], pred_test[:, 1], label='predictions', color=color[1], s=2, alpha=0.7)
    ax1.set_xlabel('v standardized')
    ax1.set_ylabel(r'$\omega$ standardized')
    title1 = 'Standardized results with confidence ellipses'
    ax1.set_title(title1)
    ax1.set_box_aspect(1)
    ax1.legend()

    ax2.scatter(test[:, 9], test[:, 10], label='ground truth', color=color[0], s=2)
    ax2.scatter(unscale_pred_test_v, unscale_pred_test_w, label='predictions', color=color[1], s=2, alpha=0.7)
    ax2.set_xlabel('v (m/s)')
    ax2.set_ylabel(r'$\omega$ (rad/s)')
    title2 = 'Prediction on test data transformed \n back to not standardized scale'
    ax2.set_title(title2)
    ax2.set_box_aspect(1)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    title = 'Prediction on test data with 9 inputs and 2 outputs \n' \
            'v MSE: ' + str("{0:.4f}".format(mse_test[0])) + \
            r', $\omega$ MSE: ' + str("{0:.4f}".format(mse_test[1])) + '\n' + \
            r'v $R^2$ score: ' + str("{0:.4f}".format(r2_test[0])) + \
            r', $\omega$ $R^2$ score: ' + str("{0:.4f}".format(r2_test[1]))
    fig.suptitle(title, fontsize=15)
    plt.legend()
    plt.savefig('figures/test_predictions_big_ellipses.png')
    plt.show()


def cross_validation_results():
    """
    Plot cross validation results
    :return:
    """
    fig, axs = plt.subplots(3, 2, figsize=(10, 11))
    fig.delaxes(axs[2, 1])
    axlist = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0]]
    models = ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']
    data_folded = load_kfolds_data('datasets/cross_valid_splits_scaled_shuffled.csv')
    mse_valid_arr = []
    r2_valid_arr = []

    for i in range(5):
        test_cv_set = data_folded[i, :, :]
        test_cv_features = test_cv_set[:, :9]
        test_cv_targets = test_cv_set[:, 9:]

        model_filename = 'models/' + models[i]
        gmr_cv = load_model(model_filename)

        print("Cross validation fold", i+1)
        start_time = time.time()
        valid_pred, valid_cov = gmr_cv.predict(test_cv_features, 9, 2)
        print("fold prediction time (s): ", time.time()-start_time)
        mse_valid, _, _, r2_valid = calculate_error_metrics(test_cv_targets, valid_pred)
        mse_valid_arr.append(mse_valid)
        r2_valid_arr.append(r2_valid)

        # extract the ellipses to draw
        yycov = gmr_cv.cov_outputs
        yymeans = gmr_cv.means_outputs

        plot_confidence_ellipse(yymeans, yycov, axlist[i], color='cyan', scale=1.0)
        axlist[i].scatter(test_cv_targets[:, 0], test_cv_targets[:, 1], label='ground truth', color=color[0], s=2)
        axlist[i].scatter(valid_pred[:, 0], valid_pred[:, 1], label='predictions', color=color[1], s=2, alpha=0.7)
        axlist[i].set_xlabel('v standardized')
        axlist[i].set_ylabel(r'$\omega$ standardized')
        title = 'Cross validation fold ' + str(i+1)
        axlist[i].set_title(title)

    mse_valid_arr = np.array(mse_valid_arr)
    r2_valid_arr = np.array(r2_valid_arr)
    mean_mse = np.mean(mse_valid_arr, axis=0)
    mean_r2 = np.mean(r2_valid_arr, axis=0)

    fig.tight_layout()
    fig.subplots_adjust(top=0.89)
    title = '5-folds cross validation, 9 inputs and 2 outputs \n' \
            'mean v MSE: ' + str("{0:.4f}".format(mean_mse[0])) + \
            r', mean $\omega$ MSE: ' + str("{0:.4f}".format(mean_mse[1])) + '\n' + \
            r'mean v $R^2$ score: ' + str("{0:.4f}".format(mean_r2[0])) + \
            r', mean $\omega$ $R^2$ score: ' + str("{0:.4f}".format(mean_r2[1]))
    fig.suptitle(title, fontsize=15)
    plt.legend()
    plt.savefig('figures/cross_validation.png')
    plt.show()


def main():
    ## load entire dataset
    dataset7in2out = np.genfromtxt('datasets/learning_dataset_0_ordered.csv', delimiter=",")
    dataset9in2out = np.genfromtxt('datasets/learning_dataset_0_bigger.csv', delimiter=",")
    # print("\n============== PART 1: JUSTIFY LEARNING AIMS ==============")
    # justify_learning_aim(dataset7in2out)
    # justify_learning_aim_with_acclim(dataset9in2out)
    # print("\n\n============== PART 2: TEST ON SIMPLE DATASET ==============")
    # test_simple_functions()
    # print("\n\n============== PART 3: K-TUNING USING SKLEARN ==============")
    # plot_bic_from_sklearn()
    # plot_bic_from_sklearn_acclim()
    print("\n\n============== PART 3: PLOT RESULTS FROM TRAINED MODELS ==============")
    plot_prediction_results_ellipse()
    plot_prediction_results_acclim()
    print("\n\n============== PART 3: PLOT RESULTS CROSS VALIDATION ==============")
    cross_validation_results()


if __name__ == "__main__":
    main()
