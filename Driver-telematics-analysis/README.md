The problem on Kaggle that I have been working on is to classify automobile drivers.  The dataset includes 2736 directories which correspond to a unique driver.  Each directory contains 200 CSV files of (x,y)-coordinates recorded every second, and these files correspond to a different drive.  A small subset of the 200 files for each driver are driving files for a different driver, and the purpose is to determine which of the 200 files in each directory correspond to the driver of interest, and compute the probability that each file is actually belonging to the driver.

The data was anonymized by randomly rotating and/or mirror images of the drivers' trajectories.  I built in an algorithm for normalizing the orientations of the trajectories by checking for mirror flips and then auto-rotating the data sets.  I built a set of 113 features which could potentially characterize each driver.  These initial features included:
```sh
•	Total trip distance
•	Average speed (including stops)
•	Max speed
•	The standard deviation of the speeds
•	The average positive acceleration
•	The average deceleration
•	Maximum positive acceleration
•	Minimum deceleration
•	The standard deviation of the acceleration
```
I also included checks for whether the driver was driving on highways, main roads (not highways) and/or back roads, as well as the average speeds and the percentage of the total trip spent in each of those situations.  There were also features for:
```sh
•	Return trip? (Boolean, check if the second half of the trip returns within an epsilon ball of the origin with a fixed radius, and a Euclidean metric)
•	Large acceleration from a stop, and the magnitude of the acceleration
•	Large deceleration to a stop, and the magnitude
The acceleration data was very noisy since it is a second order derivative of the data; I smoothed the acceleration data using a fast Fourier transform where I kept only the first 10 modes.
•	50-bin histogram of centripetal acceleration values
•	20-bin histogram of linear acceleration values
•	20-bin histogram of speed values
```

In order to classify the drivers I used a random forest with 500 trees, where at each node of the individual trees I grew the next branch by optimizing the information entropy, while checking a minimum of 5 features.  I performed a 5-fold cross-validation to validate my model.  This approach resulted in an 'Area Under the Curve' score of 0.86297 on Kaggle, and I finished the competition in the top 25% of competitors.

Some sample data visualizations can be seen below:
![alt tag](https://github.com/brainsqueeze/Kaggle-competitions/blob/master/Driver-telematics-analysis/plots/history_1-1.png)
![alt tag](https://github.com/brainsqueeze/Kaggle-competitions/blob/master/Driver-telematics-analysis/plots/velocity_1-1.png)
![alt tag](https://github.com/brainsqueeze/Kaggle-competitions/blob/master/Driver-telematics-analysis/plots/acceleration_1-1.png)
![alt tag](https://github.com/brainsqueeze/Kaggle-competitions/blob/master/Driver-telematics-analysis/plots/Speed_hist.png)
![alt tag](https://github.com/brainsqueeze/Kaggle-competitions/blob/master/Driver-telematics-analysis/plots/Accel_hist.png)
![alt tag](https://github.com/brainsqueeze/Kaggle-competitions/blob/master/Driver-telematics-analysis/plots/Cent_Accel_hist.png)