import numpy as np
import pandas as pd
import scipy
import math

from sklearn import ensemble
from sklearn import cross_validation

import time
import os


def make_features(driver_history):
    df = pd.read_csv(driver_history)
    x = pd.Series(df['x'])
    y = pd.Series(df['y'])

    if x[len(x) - 1] < 0:
        x = -x
    if y[len(y) - 1] < 0:
        y = -y

    # rotate each route WRT the origin
    theta = np.arctan(np.float64(y[len(y) - 1]) / np.float64(x[len(x) - 1]))
    x_ = np.cos(theta) * x[:] + np.sin(theta) * y[:]
    y_ = -np.sin(theta) * x[:] + np.cos(theta) * y[:]
    x, y = x_[:], y_[:]

    vel = 3.6 * np.power(np.power(np.diff(x), 2) + np.power(np.diff(y), 2), 0.5)
    x_vel, y_vel = 3.6 * np.diff(x), 3.6 * np.diff(y)
    accel = (1. / 3.6) * np.diff(vel)
    x_accel, y_accel = (1. / 3.6) * np.diff(x_vel), (1. / 3.6) * np.diff(y_vel)
    fft_x_accel, fft_y_accel = scipy.fft(x_accel), scipy.fft(y_accel)

    fft = scipy.fft(accel)
    for x in range(len(fft)):
        if x >= 50:
            fft[x] = 0
            fft_x_accel[x] = 0
            fft_y_accel[x] = 0
    inv_fft = scipy.ifft(fft).real
    inv_fft_x_accel = scipy.ifft(fft_x_accel).real
    inv_fft_y_accel = scipy.ifft(fft_y_accel).real

    cent_accel = abs(
        inv_fft_x_accel * (np.float64(y_vel[:-1]) / vel[:-1]) - inv_fft_y_accel * (np.float64(x_vel[:-1]) / vel[:-1]))
    cent_accel_hist = np.histogram(cent_accel, bins=50, range=(0, 30), density=False)[0]
    accel_ampl = np.power(np.power(inv_fft_x_accel, 2) + np.power(inv_fft_y_accel, 2), 0.5)
    accel_ampl_hist = np.histogram(accel_ampl, bins=20, range=(0, 4 * 9.8), density=False)[0]

    total_distance = np.sum(vel) / 3600  # in km
    avg_speed = np.mean(vel)  # in km/hr
    max_speed = max(vel)  # in km/hr
    var_speed = np.std(vel)  # in km/hr
    avg_accel = np.mean(inv_fft[inv_fft > 0])  # in km/hr^2
    avg_decel = np.mean(inv_fft[inv_fft < 0])  # in km/hr^2
    if len(inv_fft[abs(inv_fft) > 0]) > 0:
        max_accel = max(inv_fft[inv_fft > 0])  # in km/hr^2
        min_accel = min(inv_fft[inv_fft < 0])  # in km/hr^2
    else:
        max_accel = 0
        min_accel = 0
    std_accel = np.std(inv_fft)  # in km/hr^2
    speed_hist = np.histogram(vel, bins=20, range=(0, 130), density=False)[0]
    theta_hist = \
        np.histogram(np.arctan(np.float64(y) / np.float64(x)), bins=20, range=(-1.5708, 1.5708), density=False)[0]

    highway, mainroad, backroad = 0, 0, 0  # 0 is False, 1 is True
    hightime, maintime, backtime = 0, 0, 0
    hi_speed_avg, main_speed_avg, back_speed_avg = 0, 0, 0
    if np.all(np.diff(np.where(vel[vel > 80])) == 1):
        if len(vel[vel > 80]) > 300:
            highway = 1
            hightime = float(len(vel[vel > 80])) / len(x)
            hi_speed_avg = np.mean(vel[vel > 80])

    if np.all(np.diff(np.where(vel[vel > 80])) == 1):
        if len(vel[vel > 80]) in range(300):
            mainroad = 1
            maintime = float(len(vel[vel > 80])) / len(x)
            main_speed_avg = np.mean(vel[vel > 80])

    if np.any(np.diff(np.where(vel[vel < 80])) == 1):
        backroad = 1
        backtime = float(len(vel[vel < 80])) / len(x)
        back_speed_avg = np.mean(vel[vel < 80])

    returned = 0
    if min(np.linalg.norm(np.array([x[len(x) / 2:], y[len(y) / 2:]]))) < 150:
        returned = 1

    # High accel is more than g/2
    hi_accel, brake_accel, hi_decel, brake_decel = 0, 0, 0, 0
    if len(inv_fft[np.where(vel < 1)[0] - 1]) > 0:
        if max(inv_fft[np.where(vel < 1)[0] - 1]) > 0.5 * 9.8:
            hi_accel = 1
            brake_accel = max(abs(inv_fft[np.where(vel < 1)[0] - 1]))
        if min(inv_fft[np.where(vel < 1)[0] - 1]) < -0.5 * 9.8:
            hi_decel = 1
            brake_decel = max(abs(inv_fft[np.where(vel < 1)[0] - 1]))

    return np.append(
        [total_distance, returned, avg_speed, max_speed, var_speed, avg_accel, avg_decel, max_accel, min_accel,
         std_accel, highway,
         hightime, hi_speed_avg, mainroad, maintime, main_speed_avg, backroad, backtime, back_speed_avg, hi_accel,
         brake_accel,
         hi_decel, brake_decel], np.append(np.append(cent_accel_hist, speed_hist), accel_ampl_hist))


RefData = np.array([]).reshape(0, 113)
RefTarget = np.array([])
np.random.seed()
drivers = np.random.choice(os.listdir('drivers'), 800)
for driver in drivers:
    history = 'drivers/%s/%s.csv' % (driver, np.random.choice(np.arange(1, 201)))

    RefData = np.vstack([RefData, make_features(history)])
    RefTarget = np.r_[RefTarget, [0]]
RefData[np.isnan(RefData)] = 0
drivers = np.delete(drivers, np.s_[:])

start_time = time.time()
drivers = os.listdir('drivers')

List = np.array([]).reshape((0, 2))
count = 1
for driver in drivers[:]:
    Data = np.array([]).reshape(0, 113)
    Target = np.array([])
    DriverData = np.array([]).reshape(0, 113)
    DriverTarget = np.array([])
    for i in range(1, 201):
        history = 'drivers/%s/%s.csv' % (driver, i)

        DriverData = np.vstack([DriverData, make_features(history)])
        DriverTarget = np.r_[DriverTarget, [1]]

    DriverData[np.isnan(DriverData)] = 0

    Data = np.append(RefData, DriverData, axis=0)
    Target = np.append(RefTarget, DriverTarget)

    Data[np.isnan(Data)] = 0

    clf = ensemble.RandomForestClassifier(n_estimators=500, criterion='entropy', min_samples_leaf=5, oob_score=True)
    # scores = cross_validation.cross_val_score(clf, Data, Target, cv=5, scoring='roc_auc')
    # print scores.mean()
    clf.fit(Data, Target)

    Driver = np.array(['%s_%s' % (driver, j) for j in range(1, 201)]).reshape((200, 1))
    Prob = np.append(np.mat(Driver), np.mat(clf.predict_proba(DriverData)[:, 1]).reshape((200, 1)), axis=1)

    List = np.append(List, Prob, axis=0)
    # print 'No.: %s, Driver: %s, # True: %s, Score: %s' % (count, driver, clf.predict(DriverData)[clf.predict(DriverData) == 1].size, clf.score(DriverData, DriverTarget))
    print('No.: %s, Driver: %s' % (count, driver))
    count += 1

print('time = %s seconds' % (time.time() - start_time))
title = np.array(['driver_trip', 'prob']).reshape((1, 2))

List = np.append(title, List, axis=0)
np.savetxt('Submission.csv', List, delimiter=',', fmt='%s')
