from gmmgmr import *
import numpy as np
from sklearn.mixture import GaussianMixture
from utilities import *
from utilities import Standardize
import time


def train_original():
    print("> \t Starting training on truncated learning dataset")
    # load the entire dataset
    dataset0 = np.genfromtxt('datasets/learning_dataset_0_ordered_trunc.csv', delimiter=",")
    features0 = dataset0[:, 1:]

    # reduce the dataset0 down to 1/10th
    features0 = features0[0::10]
    features0_shuffle = features0
    np.random.shuffle(features0_shuffle)

    # K parameter search using sklearn GMM implementation
    k_list = np.arange(start=1, stop=60, step=1)
    bic_array = []
    print("\t\t k-search")
    for i in range(len(k_list)):
        gmm = GaussianMixture(n_components=k_list[i], covariance_type='full').fit(features0_shuffle)
        # print("fitted", i)
        bic_temp = gmm.bic(features0_shuffle)
        bic_array.append(bic_temp)

    # save the bic values from sklearn
    np.savetxt('temp/sklearn_bic.csv', np.array(bic_array), delimiter=',')

    # build the dataset splits for training GMM-GMR
    test = features0[0::5]
    train = np.delete(features0, np.s_[0::5], 0)

    # standardize the data
    data_temp = train[:, :]
    train_shuffle = data_temp
    np.random.shuffle(train_shuffle)
    num_train_samples = train.shape[0]
    num_test_samples = test.shape[0]

    train_scaled = np.zeros((num_train_samples, features0.shape[1]))
    train_shuffle_scaled = np.zeros((num_train_samples, features0.shape[1]))
    test_scaled = np.zeros((num_test_samples, features0.shape[1]))
    scaler_list = []
    for i in range(features0.shape[1]):
        scaler = Standardize().fit(train[:, i])
        scaler_list.append(scaler)
        train_scaled[:, i] = np.reshape(scaler.transform(train[:, i]), (num_train_samples, ))
        train_shuffle_scaled[:, i] = np.reshape(scaler.transform(train_shuffle[:, i]), (num_train_samples, ))
        test_scaled[:, i] = np.reshape(scaler.transform(test[:, i]), (num_test_samples, ))

    # save this version of dataset
    np.savetxt("temp/train_shuffle_ordered.csv", train_shuffle, delimiter=",")
    np.savetxt("temp/test_ordered.csv", test, delimiter=",")

    print("\t\t Start training on original dataset")
    gmr_shuffle_ds0 = GMMGMR(20)
    start_time = time.time()
    gmr_shuffle_ds0.fit(train_shuffle_scaled)
    print("training time: ", time.time()-start_time)
    save_model(gmr_shuffle_ds0, 'temp/gmrds0_shuffle_model')

    test_scaled_features = test[:, :7]
    test_scaled_targets = test[:, 7:]

    start_time = time.time()
    pred_test_sh, pred_test_cov_sh = gmr_shuffle_ds0.predict(test_scaled_features, 7, 2)
    print("prediction time: ", time.time()-start_time)
    mse, mae, r2 = calculate_error_metrics(test_scaled_targets, pred_test_sh)
    print("mse", mse)
    print("mae", mae)
    print("r2 score", r2)
    return


def train_expanded():
    print("> \t Starting training on truncated expanded dataset")
    # load the entire dataset
    dataset0 = np.genfromtxt('datasets/learning_dataset_0_bigger_trunc.csv', delimiter=",")
    features0 = dataset0[:, 1:]

    # reduce the dataset0 down to 1/10th
    features0 = features0[0::10]
    features0_shuffle = features0
    np.random.shuffle(features0_shuffle)

    # K parameter search using sklearn GMM implementation
    k_list = np.arange(start=1, stop=60, step=1)
    bic_array = []
    print("\t\t k-search")
    for i in range(len(k_list)):
        gmm = GaussianMixture(n_components=k_list[i], covariance_type='full').fit(features0_shuffle)
        # print("fitted", i)
        bic_temp = gmm.bic(features0_shuffle)
        bic_array.append(bic_temp)

    # save the bic values from sklearn
    np.savetxt('temp/sklearn_bic_big.csv', np.array(bic_array), delimiter=',')

    # build the dataset splits for training GMM-GMR
    test = features0[0::5]
    train = np.delete(features0, np.s_[0::5], 0)

    # standardize the data
    data_temp = train[:, :]
    train_shuffle = data_temp
    np.random.shuffle(train_shuffle)
    num_train_samples = train.shape[0]
    num_test_samples = test.shape[0]

    train_scaled = np.zeros((num_train_samples, features0.shape[1]))
    train_shuffle_scaled = np.zeros((num_train_samples, features0.shape[1]))
    test_scaled = np.zeros((num_test_samples, features0.shape[1]))
    scaler_list = []
    for i in range(features0.shape[1]):
        scaler = Standardize().fit(train[:, i])
        scaler_list.append(scaler)
        train_scaled[:, i] = np.reshape(scaler.transform(train[:, i]), (num_train_samples, ))
        train_shuffle_scaled[:, i] = np.reshape(scaler.transform(train_shuffle[:, i]), (num_train_samples, ))
        test_scaled[:, i] = np.reshape(scaler.transform(test[:, i]), (num_test_samples, ))

    # save this version of dataset
    np.savetxt("temp/train_shuffle_big.csv", train_shuffle, delimiter=",")
    np.savetxt("temp/test_big.csv", test, delimiter=",")

    print("\t\t Start training on expanded dataset")
    gmr_shuffle_ds0 = GMMGMR(15)
    start_time = time.time()
    gmr_shuffle_ds0.fit(train_shuffle_scaled)
    print("training time: ", time.time()-start_time)
    save_model(gmr_shuffle_ds0, 'temp/gmrds0_big_shuffle_model')

    test_scaled_features = test[:, :9]
    test_scaled_targets = test[:, 9:]

    start_time = time.time()
    pred_test_sh, pred_test_cov_sh = gmr_shuffle_ds0.predict(test_scaled_features, 9, 2)
    print("prediction time: ", time.time()-start_time)
    mse, mae, r2 = calculate_error_metrics(test_scaled_targets, pred_test_sh)
    print("mse", mse)
    print("mae", mae)
    print("r2 score", r2)


def run_cross_validation():
    print("> \t Starting Cross Validation")
    # load the entire dataset
    dataset0 = np.genfromtxt('datasets/learning_dataset_0_bigger_trunc.csv', delimiter=",")
    features0 = dataset0[:, 1:]

    # standardize features0 for kfolds
    scaler_cv_list = []
    features0_scaled = np.empty(features0.shape)
    for i in range(features0.shape[1]):
        scaler_cv = Standardize().fit(features0[:, i])
        scaler_cv_list.append(scaler_cv)
        features0_scaled[:, i] = np.reshape(scaler_cv.transform(features0[:, i]), (features0.shape[0], ))
    np.random.shuffle(features0_scaled)

    data_folded = kfolds_cross_validation(features0_scaled)
    file_name = 'temp/cross_valid_splits_scaled_shuffled'
    cv_splits = data_folded.tolist()
    with open(file_name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(cv_splits)

    means_filename = ['cv_means_1.csv', 'cv_means_2.csv', 'cv_means_3.csv', 'cv_means_4.csv', 'cv_means_5.csv']
    cov_filename = ['cv_cov_1.csv', 'cv_cov_2.csv', 'cv_cov_3.csv', 'cv_cov_4.csv', 'cv_cov_5.csv']

    # loop through the folds to construct training and testing sets
    # and do training and testing
    # save trained models for later analysis
    for i in range(5):
        idx = np.arange(5)
        train_idx = np.delete(idx, i, 0)
        train_cv_set = np.concatenate((data_folded[train_idx, :, :]), axis=0)
        test_cv_set = data_folded[i, :, :]
        test_cv_features = test_cv_set[:, :9]
        test_cv_targets = test_cv_set[:, 9:]

        gmr_cv = GMMGMR(15)
        print("training fold", i)
        start_time = time.time()
        gmr_cv.fit(train_cv_set)
        print("training time: ", time.time()-start_time)

        cov_file_name = 'temp/' + cov_filename[i]
        model_cov = gmr_cv.covariances
        model_cov = model_cov.tolist()
        with open(cov_file_name+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(model_cov)

        means_file_name = 'temp/' + means_filename[i]
        model_means = gmr_cv.means
        model_means = model_means.tolist()
        with open(means_file_name+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(model_means)

        start_time = time.time()
        valid_pred, valid_cov = gmr_cv.predict(test_cv_features, 9, 2)
        print("prediction time (s): ", (time.time()-start_time))
        mse, mae, r2 = calculate_error_metrics(test_cv_targets, valid_pred)
        print("mse", mse)
        print("mae", mae)
        print("r2 score", r2)


def train_mm():

    return


def main():
    # train_original()
    # train_expanded()
    ## cross validation takes forever, proceed with caution
    # run_cross_validation()

    train_mm()


if __name__ == "__main__":
    main()
