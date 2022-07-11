"""
Parser for the dataset, outputs each data file to a numpy array
"""
import numpy as np
import pandas as pd
from utilities import *
import csv


def Odometry_loader(filename, calc_duration=True):
    """
    Import the odometry data into a numpy array and calculate the duration of
    velocity commands issued
    Check for time duration = 0 and delete those entries

    :param filename: path to the .dat file
    :param calc_duration: option to compute control duration
    :return: Nx4 array, columns are time, forward v, angular v, duration
    """
    odom_data = np.loadtxt(filename)
    if calc_duration:
        # Calculate the duration for each velocity command
        # We assume that issued velocity commands are constant over the time duration between
        #  the logged command and the next command
        # for the last control command, assume it was issued for the duration of the average control frequency
        #  from http://asrl.utias.utoronto.ca/datasets/mrclam/index.html this is ~67Hz
        duration_arr = np.zeros((odom_data.shape[0], 1))
        duration_arr[-1] = 1 / 67
        for i in range(odom_data.shape[0] - 1):
            duration_arr[i] = odom_data[i+1, 0] - odom_data[i, 0]
        odom_data = np.concatenate((odom_data, duration_arr), axis=1)
        zero_idx = np.squeeze(np.where(duration_arr[:, 0] == 0))
        odom_data = np.delete(odom_data, zero_idx, 0)
        return odom_data
    else:
        return odom_data


def load_kfolds_data(filename):
    """
    Load the k-fold split data

    :param filename: string
    :return: numpy array of the data folds
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data_read = list(reader)

    folded_data = []
    for row in data_read:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        folded_data.append(nwrow)
    return np.array(folded_data)


def interpolate_groundtruth_prev_command(controls, groundtruth, reduce_dim=False):
    """
    Interpolate the groundtruth robot poses to the control command timesteps
    Include previous timestep in the dataset

    :param controls: [v, omega, dt]
    :param groundtruth: parsed from robot dataset
    :param reduce_dim: flag to return the delta pose instead (depracated)
    :return: dataset
    """
    intpl_gt_start_arr = np.zeros((1, 4))
    intpl_gt_end_arr = np.zeros((1, 4))
    prev_command_arr = np.zeros((1, 2))
    # print(controls.shape[0])

    for i in range(controls.shape[0] - 1):
        # print("control number", i)
        time_start = controls[i, 0]
        time_end = controls[i+1, 0]
        # find the relevant data entries in groundtruth
        # print(time_start)
        # print(groundtruth[:, 0:10])
        prev_t_idx1 = np.squeeze(np.where(groundtruth[:, 0] <= time_start))
        next_t_idx1 = np.squeeze(np.where(groundtruth[:, 0] >= time_start))
        # print(type(prev_t_idx1))
        # print(prev_t_idx1.size)
        if prev_t_idx1.size == 1:
            prev_t_idx1 = prev_t_idx1.item()
            # print('in isscalar')
        else:
            # print('in else')
            # print(prev_t_idx1)
            prev_t_idx1 = prev_t_idx1[-1]
        if next_t_idx1.size == 1:
            next_t_idx1 = next_t_idx1.item()
        else:
            next_t_idx1 = next_t_idx1[0]

        if prev_t_idx1 == next_t_idx1:
            intpl_gt_start = groundtruth[prev_t_idx1, :]
        else:
            ratio1 = (time_start - groundtruth[prev_t_idx1, 0]) / (groundtruth[next_t_idx1, 0] - groundtruth[prev_t_idx1, 0])
            intpl_gt_start = ratio1 * (groundtruth[next_t_idx1, :] - groundtruth[prev_t_idx1, :]) + groundtruth[prev_t_idx1, :]

        intpl_gt_start_arr = np.concatenate((intpl_gt_start_arr, np.array([intpl_gt_start])), axis=0)

        prev_t_idx2 = np.squeeze(np.where(groundtruth[:, 0] <= time_end))
        next_t_idx2 = np.squeeze(np.where(groundtruth[:, 0] >= time_end))
        if prev_t_idx2.size == 1:
            prev_t_idx2 = prev_t_idx2.item()
        else:
            prev_t_idx2 = prev_t_idx2[-1]
        if next_t_idx2.size == 1:
            next_t_idx2 = next_t_idx2.item()
        else:
            next_t_idx2 = next_t_idx2[0]

        if prev_t_idx2 == next_t_idx2:
            intpl_gt_end = groundtruth[prev_t_idx2, :]
        else:
            ratio2 = (time_end - groundtruth[prev_t_idx2, 0]) / (groundtruth[next_t_idx2, 0] - groundtruth[prev_t_idx2, 0])
            intpl_gt_end = ratio2 * (groundtruth[next_t_idx2, :] - groundtruth[prev_t_idx2, :]) + groundtruth[prev_t_idx2, :]

        intpl_gt_end_arr = np.concatenate((intpl_gt_end_arr, np.array([intpl_gt_end])), axis=0)

    prev_command_arr = np.concatenate((prev_command_arr, controls[:-2, 1:3]))

    intpl_gt_start_arr = np.delete(intpl_gt_start_arr, 0, 0)
    intpl_gt_end_arr = np.delete(intpl_gt_end_arr, 0, 0)    # strip first row
    intpl_gt_end_arr = np.delete(intpl_gt_end_arr, 0, 1)    # strip first column

    if reduce_dim:
        gt_inputs = intpl_gt_end_arr[:, 1:] - intpl_gt_start_arr[:, 1:]
        gt_arr = np.concatenate((intpl_gt_start_arr[:, 0:1], gt_inputs, controls[:-1, 1:]), axis=1)
    else:
        gt_arr = np.concatenate((intpl_gt_start_arr, intpl_gt_end_arr, prev_command_arr, controls[:-1, 3:], controls[:-1, 1:3]), axis=1)
    return gt_arr


def interpolate_groundtruth_ordered(controls, groundtruth, reduce_dim=False):
    """
    Interpolate the groundtruth robot poses to the control command timesteps

    :param controls: [v, omega, dt]
    :param groundtruth: parsed from robot dataset
    :param reduce_dim: flag to return the delta pose instead (depracated)
    :return: dataset
    """
    intpl_gt_start_arr = np.zeros((1, 4))
    intpl_gt_end_arr = np.zeros((1, 4))
    # print(controls.shape[0])

    for i in range(controls.shape[0] - 1):
        # print("control number", i)
        time_start = controls[i, 0]
        time_end = controls[i+1, 0]
        # find the relevant data entries in groundtruth
        # print(time_start)
        # print(groundtruth[:, 0:10])
        prev_t_idx1 = np.squeeze(np.where(groundtruth[:, 0] <= time_start))
        next_t_idx1 = np.squeeze(np.where(groundtruth[:, 0] >= time_start))
        # print(type(prev_t_idx1))
        # print(prev_t_idx1.size)
        if prev_t_idx1.size == 1:
            prev_t_idx1 = prev_t_idx1.item()
            # print('in isscalar')
        else:
            # print('in else')
            # print(prev_t_idx1)
            prev_t_idx1 = prev_t_idx1[-1]
        if next_t_idx1.size == 1:
            next_t_idx1 = next_t_idx1.item()
        else:
            next_t_idx1 = next_t_idx1[0]

        if prev_t_idx1 == next_t_idx1:
            intpl_gt_start = groundtruth[prev_t_idx1, :]
        else:
            ratio1 = (time_start - groundtruth[prev_t_idx1, 0]) / (groundtruth[next_t_idx1, 0] - groundtruth[prev_t_idx1, 0])
            intpl_gt_start = ratio1 * (groundtruth[next_t_idx1, :] - groundtruth[prev_t_idx1, :]) + groundtruth[prev_t_idx1, :]

        intpl_gt_start_arr = np.concatenate((intpl_gt_start_arr, np.array([intpl_gt_start])), axis=0)

        prev_t_idx2 = np.squeeze(np.where(groundtruth[:, 0] <= time_end))
        next_t_idx2 = np.squeeze(np.where(groundtruth[:, 0] >= time_end))
        if prev_t_idx2.size == 1:
            prev_t_idx2 = prev_t_idx2.item()
        else:
            prev_t_idx2 = prev_t_idx2[-1]
        if next_t_idx2.size == 1:
            next_t_idx2 = next_t_idx2.item()
        else:
            next_t_idx2 = next_t_idx2[0]

        if prev_t_idx2 == next_t_idx2:
            intpl_gt_end = groundtruth[prev_t_idx2, :]
        else:
            ratio2 = (time_end - groundtruth[prev_t_idx2, 0]) / (groundtruth[next_t_idx2, 0] - groundtruth[prev_t_idx2, 0])
            intpl_gt_end = ratio2 * (groundtruth[next_t_idx2, :] - groundtruth[prev_t_idx2, :]) + groundtruth[prev_t_idx2, :]

        intpl_gt_end_arr = np.concatenate((intpl_gt_end_arr, np.array([intpl_gt_end])), axis=0)

    intpl_gt_start_arr = np.delete(intpl_gt_start_arr, 0, 0)
    intpl_gt_end_arr = np.delete(intpl_gt_end_arr, 0, 0)    # strip first row
    intpl_gt_end_arr = np.delete(intpl_gt_end_arr, 0, 1)    # strip first column

    if reduce_dim:
        gt_inputs = intpl_gt_end_arr[:, 1:] - intpl_gt_start_arr[:, 1:]
        gt_arr = np.concatenate((intpl_gt_start_arr[:, 0:1], gt_inputs, controls[:-1, 1:]), axis=1)
    else:
        gt_arr = np.concatenate((intpl_gt_start_arr, intpl_gt_end_arr, controls[:-1, 3:], controls[:-1, 1:3]), axis=1)
    return gt_arr


def create_learning_dataset():
    ds0_odom = Odometry_loader('ds0/ds0_Odometry.dat')
    ds0_robotgt = np.loadtxt('ds0/ds0_Groundtruth.dat')

    learning_dataset0_big = interpolate_groundtruth_prev_command(ds0_odom, ds0_robotgt)
    learning_dataset0 = interpolate_groundtruth_ordered(ds0_odom, ds0_robotgt)

    learning_dataset0_big_trunc = learning_dataset0_big[:1000, :]
    learning_dataset0_trunc = learning_dataset0[:1000, :]

    colnames_big = ['t', 'x', 'y', 'heading', 'x_next', 'y_next', 'heading_next', 'v_prev', 'w_prev', 'dt', 'v', 'w']
    colnames = ['t', 'x', 'y', 'heading', 'x_next', 'y_next', 'heading_next', 'v', 'w', 'dt']

    df0_big = pd.DataFrame(learning_dataset0_big, columns=colnames_big)
    df0 = pd.DataFrame(learning_dataset0, columns=colnames)
    df0_big.to_csv('datasets/learning_dataset_0_bigger.csv', index=False, header=False)
    df0.to_csv('datasets/learning_dataset_0_ordered.csv', index=False, header=False)

    df0_big_trunc = pd.DataFrame(learning_dataset0_big_trunc, columns=colnames_big)
    df0_trunc = pd.DataFrame(learning_dataset0_trunc, columns=colnames)
    df0_big_trunc.to_csv('datasets/learning_dataset_0_bigger_trunc.csv', index=False, header=False)
    df0_trunc.to_csv('datasets/learning_dataset_0_ordered_trunc.csv', index=False, header=False)


def main():
    create_learning_dataset()


if __name__ == "__main__":
    main()
