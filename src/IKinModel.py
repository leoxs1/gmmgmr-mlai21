import numpy as np
import math
from utilities import *


def inverse_kinematics_acclim(current_pose, next_pose, prev_command, dt, threshold=1e-5):
    """
    IKin model with acceleration limits imposed

    :param current_pose:
    :param next_pose:
    :param prev_command:
    :param dt:
    :param threshold:
    :return: required controls
    """
    max_linacc = 0.288
    max_angacc = 5.579
    acc_dt = 0.01

    omega = np.squeeze((normalize_angle(next_pose[2]) - normalize_angle(current_pose[2])) / dt)

    # Check if controls are within the specified acceleration limits
    angacc = (omega - prev_command[1]) / acc_dt
    if abs(angacc) >= max_angacc:
        omega = np.sign(angacc) * max_angacc * acc_dt + prev_command[1]

    linear = 0
    if dt < threshold:
        print("too small dt: ", dt)
    if abs(omega) < threshold:
        omega = 0
        if ( -threshold < normalize_angle(current_pose[2]) < threshold or \
                (abs(normalize_angle(current_pose[2])) - np.pi < threshold) ):
            linear = (next_pose[0] - current_pose[0]) / (dt * np.cos(current_pose[2]))

        elif (abs(normalize_angle(current_pose[2])) - np.pi/2) < threshold:
            linear = (next_pose[1] - current_pose[1]) / (dt * np.sin(current_pose[2]))
        else:
            linear_fromx = (next_pose[0] - current_pose[0]) / (dt * np.cos(current_pose[2]))
            linear_fromy = (next_pose[1] - current_pose[1]) / (dt * np.sin(current_pose[2]))
            # choose the positive one
            # print("with 2 possible linears", linear_fromx, linear_fromy)
            linear = max([linear_fromx, linear_fromy])

    else:
        sindiff = np.sin(current_pose[2]) - np.sin(next_pose[2])
        cosdiff = np.cos(current_pose[2]) - np.cos(next_pose[2])
        headdiff = next_pose[0] - current_pose[0]
        if abs(dt * sindiff) < threshold:
            linear = (headdiff * (next_pose[1] - current_pose[1])) / (dt * cosdiff)
        elif abs(dt * cosdiff) < threshold:
            linear = (-headdiff * (next_pose[0] - current_pose[0])) / (dt * sindiff)
        else:
            linear_fromx = (-headdiff * (next_pose[0] - current_pose[0])) / (dt * sindiff)
            linear_fromy = (headdiff * (next_pose[1] - current_pose[1])) / (dt * cosdiff)
            # print("with 2 possible linears", linear_fromx, linear_fromy)
            linear = max([linear_fromx, linear_fromy])

    # Check if controls are within the specified acceleration limits
    linacc = (linear - prev_command[0]) / acc_dt
    if abs(linacc) >= max_linacc:
        linear = np.sign(linacc) * max_linacc * acc_dt + prev_command[0]

    return np.array([[np.squeeze(linear), omega]])


def inverse_kinematics(current_pose, next_pose, dt, threshold=1e-5):
    """
    Naive IKin model, no acceleration limits

    :param current_pose:
    :param next_pose:
    :param dt:
    :param threshold:
    :return: required control
    """
    omega = np.squeeze((normalize_angle(next_pose[2]) - normalize_angle(current_pose[2])) / dt)
    linear = 0
    if dt < threshold:
        print("too small dt: ", dt)
    if abs(omega) < threshold:
        omega = 0
        if ( -threshold < normalize_angle(current_pose[2]) < threshold or \
                (abs(normalize_angle(current_pose[2])) - np.pi < threshold) ):
            linear = (next_pose[0] - current_pose[0]) / (dt * np.cos(current_pose[2]))

        elif (abs(normalize_angle(current_pose[2])) - np.pi/2) < threshold:
            linear = (next_pose[1] - current_pose[1]) / (dt * np.sin(current_pose[2]))
        else:
            linear_fromx = (next_pose[0] - current_pose[0]) / (dt * np.cos(current_pose[2]))
            linear_fromy = (next_pose[1] - current_pose[1]) / (dt * np.sin(current_pose[2]))
            # choose the positive one
            # print("with 2 possible linears", linear_fromx, linear_fromy)
            linear = max([linear_fromx, linear_fromy])

    else:
        sindiff = np.sin(current_pose[2]) - np.sin(next_pose[2])
        cosdiff = np.cos(current_pose[2]) - np.cos(next_pose[2])
        headdiff = next_pose[0] - current_pose[0]
        if abs(dt * sindiff) < threshold:
            linear = (headdiff * (next_pose[1] - current_pose[1])) / (dt * cosdiff)
        elif abs(dt * cosdiff) < threshold:
            linear = (-headdiff * (next_pose[0] - current_pose[0])) / (dt * sindiff)
        else:
            linear_fromx = (-headdiff * (next_pose[0] - current_pose[0])) / (dt * sindiff)
            linear_fromy = (headdiff * (next_pose[1] - current_pose[1])) / (dt * cosdiff)
            # print("with 2 possible linears", linear_fromx, linear_fromy)
            linear = max([linear_fromx, linear_fromy])

    return np.array([[np.squeeze(linear), omega]])
