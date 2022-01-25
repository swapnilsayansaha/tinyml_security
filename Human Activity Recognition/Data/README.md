# Auritus Activity Dataset - Activity Detection Using Earables

- This folder contains human activity data for 9 classes (walking, jogging, jumping, standing, sitting, tuning left, turning right, laying down, and falling) of basic ADL. 
- The data was collected using ear mounted inertial measurement units.
- The first three columns of each ```.csv``` file provide the accelerometer readings (in g), the next three columns provide gyroscope readings (in deg/s).
- The seventh column correspond to timestamp (in sec.)
- The last character before each ```.csv``` file corresponds to the activity: F - Falling, J - Jumping, R - Running/Jogging, Si - Sitting, St: Standing, Tl: Turning Left, Tr: Turning Right, W: Walking.
- The number in each file name (e.g., 1_F.csv) corresponds to participant number. We had 45 participants in the dataset. 
- The sampling rate was set to 100 Hz, however, there is sampling rate jitter and missing data in the dataset due to packet drops and timestamp misalignment. A low-pass filter of 5 Hz was used, and the accelerometer and gyroscope ranges were set to +- 4g and +- 500 deg/s. The BLE advertisement and connection intervals were set to 45-55 mS and 20-30 mS.






