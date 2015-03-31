import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
import os
import time

df = pd.read_csv('drivers/1/1.csv')
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

# make plots
plt.plot(X, Y)
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.savefig('history_1-1.png')
plt.clf()

plt.plot(vel)
plt.xlabel('Time [s]')
plt.ylabel('Speed [km/hr]')
plt.savefig('velocity_1-1.png')
plt.clf()

plt.plot(accel)
plt.plot(iFFT)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.savefig('acceleration_1-1.png')
plt.clf()

plt.plot(Acent)
plt.xlabel('Time [s]')
plt.ylabel('Centripetal Acceleration [m/s^2]')
plt.savefig('cent_accel_1-1.png')
plt.clf()

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

# make histograms
x, bins = np.histogram(vel, bins=20, range=(0,130), density=False)
plt.hist(vel, bins, histtype='stepfilled', facecolor='green', alpha=0.5)
plt.xlabel('Speed [km/hr]')
plt.savefig('Speed_hist.png')
plt.clf()

x, bins = np.histogram(A, bins=20, range=(0,0.2*9.8), density=False)
plt.hist(A, bins, histtype='stepfilled', facecolor='green', alpha=0.5)
plt.xlabel('Acceleration [m/s^2]')
plt.savefig('Accel_hist.png')
plt.clf()

x, bins = np.histogram(Acent, bins=50, range=(0,0.2*9.8), density=False)
plt.hist(Acent, bins, histtype='stepfilled', facecolor='green', alpha=0.5)
plt.xlabel('Centripetal Acceleration [m/s^2]')
plt.savefig('Cent_Accel_hist.png')
plt.clf()


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