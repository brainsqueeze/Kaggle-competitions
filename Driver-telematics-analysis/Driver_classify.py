import numpy as np
import pandas as pd
import scipy
import math
from sklearn import ensemble
from sklearn import cross_validation
import time

def MakeFeat(history):
    df = pd.read_csv(history)
    X = pd.Series(df['x'])
    Y = pd.Series(df['y'])

    if X[len(X)-1] < 0:
        X = -X
    if Y[len(Y)-1] < 0:
        Y = -Y

    # rotate each route WRT the origin
    theta = np.arctan(np.float64(Y[len(Y)-1])/np.float64(X[len(X)-1]))
    Xtemp = np.cos(theta)*X[:] + np.sin(theta)*Y[:]
    Ytemp = -np.sin(theta)*X[:] + np.cos(theta)*Y[:]
    X, Y = Xtemp[:], Ytemp[:]
        
    vel = 3.6*np.power(np.power(np.diff(X),2) + np.power(np.diff(Y),2), 0.5)
    VelX, VelY = 3.6*np.diff(X), 3.6*np.diff(Y)
    accel = (1./3.6)*np.diff(vel)
    Ax, Ay = (1./3.6)*np.diff(VelX), (1./3.6)*np.diff(VelY)
    FFTax, FFTay = scipy.fft(Ax), scipy.fft(Ay)
    
    FFT = scipy.fft(accel)
    for x in range(len(FFT)):
        if x >= 50:
            FFT[x]=0
            FFTax[x]=0
            FFTay[x]=0
    iFFT = scipy.ifft(FFT).real
    iFFTax = scipy.ifft(FFTax).real
    iFFTay = scipy.ifft(FFTay).real

    Acent = abs(iFFTax*(np.float64(VelY[:-1])/vel[:-1]) - iFFTay*(np.float64(VelX[:-1])/vel[:-1]))
    AChist = np.histogram(Acent, bins=50, range=(0,30), density=False)[0]
    A = np.power(np.power(iFFTax,2) + np.power(iFFTay,2), 0.5)
    Ahist = np.histogram(A, bins=20, range=(0,4*9.8), density=False)[0]

    TotDist = np.sum(vel)/3600    # in km
    AvgSpeed = np.mean(vel)       # in km/hr
    MaxSpeed = max(vel)           # in km/hr
    StdSpeed = np.std(vel)        # in km/hr
    AvgAccel = np.mean(iFFT[iFFT > 0])      # in km/hr^2
    AvgDecel = np.mean(iFFT[iFFT < 0])      # in km/hr^2
    if len(iFFT[abs(iFFT) > 0]) > 0:
        MaxAccel = max(iFFT[iFFT > 0])# in km/hr^2
        MinAccel = min(iFFT[iFFT < 0])# in km/hr^2
    else:
        MaxAccel = 0
        MinAccel = 0
    StdAccel = np.std(iFFT)       # in km/hr^2
    SpeedHist = np.histogram(vel, bins=20, range=(0,130), density=False)[0]
    ThetaHist = np.histogram(np.arctan(np.float64(Y)/np.float64(X)), bins=20, range=(-1.5708, 1.5708), density=False)[0]
    
    highway, mainroad, backroad = 0, 0, 0    # 0 is False, 1 is True
    hightime, maintime, backtime = 0, 0, 0
    hiSpeedAvg, mainSpeedAvg, backSpeedAvg = 0, 0, 0
    if np.all(np.diff(np.where(vel[vel > 80])) == 1):
        if len(vel[vel > 80]) > 300:
            highway = 1
            hightime = float(len(vel[vel > 80]))/len(X)
            hiSpeedAvg = np.mean(vel[vel > 80])
        
    if np.all(np.diff(np.where(vel[vel > 80])) == 1):
        if len(vel[vel > 80]) in range(300):
            mainroad = 1
            maintime = float(len(vel[vel > 80]))/len(X)
            mainSpeedAvg = np.mean(vel[vel > 80])

    if np.any(np.diff(np.where(vel[vel < 80])) == 1):
        backroad = 1
        backtime = float(len(vel[vel < 80]))/len(X)
        backSpeedAvg = np.mean(vel[vel < 80])

    Returned = 0
    if min(np.linalg.norm(np.array([X[len(X)/2:], Y[len(Y)/2:]]))) < 150:
        Returned = 1

    # High accel is more than g/2
    HiAccel, BrakeAccel, HiDecel, BrakeDecel = 0, 0, 0, 0
    if len(iFFT[np.where(vel < 1)[0]-1]) > 0:
        if max(iFFT[np.where(vel < 1)[0]-1]) > 0.5*9.8:
            HiAccel = 1
            BrakeAccel = max(abs(iFFT[np.where(vel < 1)[0]-1]))
        if min(iFFT[np.where(vel < 1)[0]-1]) < -0.5*9.8:
            HiDecel = 1
            BrakeDecel = max(abs(iFFT[np.where(vel < 1)[0]-1]))

    return np.append([TotDist, Returned, AvgSpeed, MaxSpeed, StdSpeed, AvgAccel, AvgDecel, MaxAccel, MinAccel, StdAccel, highway, hightime, hiSpeedAvg, mainroad, maintime, mainSpeedAvg, backroad, backtime, backSpeedAvg, HiAccel, BrakeAccel, HiDecel, BrakeDecel], np.append(np.append(AChist,SpeedHist), Ahist))


RefData = np.array([]).reshape(0, 113)
RefTarget = np.array([])
np.random.seed()
drivers = np.random.choice(os.listdir('drivers'), 800)
for driver in drivers:
    history = 'drivers/%s/%s.csv' % (driver, np.random.choice(np.arange(1,201)))
        
    RefData = np.vstack([RefData, MakeFeat(history)])
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
  
        DriverData = np.vstack([DriverData, MakeFeat(history)])
        DriverTarget = np.r_[DriverTarget, [1]]

    DriverData[np.isnan(DriverData)] = 0
            
    Data = np.append(RefData, DriverData, axis=0)
    Target = np.append(RefTarget, DriverTarget)

    Data[np.isnan(Data)] = 0
     
    clf = ensemble.RandomForestClassifier(n_estimators=500, criterion='entropy', min_samples_leaf=5, oob_score=True)
    #scores = cross_validation.cross_val_score(clf, Data, Target, cv=5, scoring='roc_auc')
    #print scores.mean()
    clf.fit(Data,Target)
 
    Driver = np.array(['%s_%s' % (driver, j) for j in range(1, 201)]).reshape((200, 1))
    Prob = np.append(np.mat(Driver), np.mat(clf.predict_proba(DriverData)[:,1]).reshape((200,1)), axis=1)

    List = np.append(List, Prob, axis=0)
    #print 'No.: %s, Driver: %s, # True: %s, Score: %s' % (count, driver, clf.predict(DriverData)[clf.predict(DriverData) == 1].size, clf.score(DriverData, DriverTarget))
    print 'No.: %s, Driver: %s' % (count, driver)  
    count += 1


print 'time = %s seconds' % (time.time() - start_time)
title = np.array(['driver_trip', 'prob']).reshape((1, 2))

List = np.append(title, List, axis=0)
np.savetxt('Submission.csv', List, delimiter=',', fmt='%s')
