# Invariance to Orientation of Wearable Motion Sensors in Automated Detection and Evaluation of Physical Therapy Exercises based on Templates

## Introduction

My name is Mats Fockaert, and I am a student at KU Leuven, in the Department of Computer Science. My thesis is focused on developing a novel algorithm for automated detection and evaluation of physical therapy exercises using wearable motion sensors, with a specific focus on obtaining invariance to orientation of the sensors.

## Problem

The use of wearable motion sensors in physical therapy has the potential to improve the effectiveness and efficiency of the therapy. However, a major challenge in using these sensors is the variability in sensor orientation, which can lead to significant differences in the data collected. This makes it difficult to accurately detect and evaluate exercises, as the same exercise performed with different sensor orientations may produce vastly different data. Previous studies in this field have been conducted, but these use segmented training data. This makes it easier to create robustness. In this work, there will be non-segmented data; one long execution of exercises.

## Obtaining Invariance to Orientation with SVD

To address this challenge, my thesis proposes using Singular Value Decomposition (SVD) to obtain invariance to sensor orientation. SVD is a powerful mathematical tool that can be used to decompose a matrix into its fundamental components, and is particularly useful for data with high variability. By applying SVD to the sensor data, we can extract the most important features of the data and eliminate the effects of sensor orientation. This is a well-known technique and is used a lot in similar work. Note, related work mostly focuses on *segmented* data. This thesis does not rely on these simplifications of the problem. The data that is used in this research does not have any indication of when an exercise execution starts, stops or what execution is correct. This seems like a small change, but proves to be a non-trivial task to handle this extra level of difficulty. For this, different approaches have to be taken.

## MTMM-DTW Algorithm

The next step in the process is to use the novel algorithm MTMM-DTW (Multi-Template Multi Match Dynamic Time Warping) for finding similarities between a test sequence of physical therapy exercises and multiple templates of each physical therapy exercise. The algorithm uses the Dynamic Time Warping (DTW) technique to compare sequences, but with the added step of comparing the test sequence to multiple templates of each exercise. The MTMM-DTW algorithm is an existing algorithm and is improved upon to tackle the *non-segmented* data. This novel approaches fuses the new OIT using SVD with some changes with MTMM-DTW in such a way that it can handle different orientation changes, sensor drift and sudden orientation changes in the sensor data.

## Conclusion

By using some form of SVD to obtain invariance to sensor orientation and the optimized MTMM-DTW algorithm for automated detection and evaluation of exercises, my thesis shows an improvement in the accuracy and efficiency of physical therapy exercise detection using wearable motion sensors in realistic scenario's. The results of this research have the potential to greatly benefit patients, therapists, and the healthcare system as a whole.

## The Code

Library with all classes used to perform research for attaining robustness to orientation and improving automated detection of subsequences in physical therapy time series data in these settings. The library is written in python and uses `jit` methods for improving certain functions with the `nopython` parameter.

This research focused on the `Physical Therapy Exercises Dataset' (Yurtman, Barshan, [2022](https://doi.org/10.24432/C5JK60)). This dataset is also used in all examples and contains time series data recorded during training sessions of 5 subjects performing 8 exercises wearing 5 sensors.

## Examples

### Loading the data

```python

from thesis import loading as Loading

l = Loading('folder_with_data')
l.load_all()
subjects = l.time_series
# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[0][1][1]

```

### Orientation changes

#### Simulating sensor orientation change

rotate the time series using: x-rotation: 64 degrees, y-rotation: 23 degrees, z-rotation: 22 degrees

```python

from thesis.oit import simulate_rotation as rotate
rotated_time_series = rotate(time_series, [64,23,22])

```

#### Simulating sudden sensor orientation changes

rotate subsequences in the time series (of length 5000) using:

- time step: 600 to 1600 with x-rotation: 64 degrees, y-rotation: 23 degrees, z-rotation: 22 degrees
- time step: 1600 to 2400 with x-rotation: 15 degrees, y-rotation: 123 degrees, z-rotation: 18 degrees
- time step: 2400 to 5000 with x-rotation: 105 degrees, y-rotation: 80 degrees, z-rotation: 150 degrees

```python

from thesis.oit import simulate_sudden_changes as rotate_sequences
rotated_time_series = rotate(time_series, [[600,1600],[1600,2400],[2400,5000]],[[64,23,22],[15,123,18],[105,80,150]])

```

### Creating robustness

#### Baseline: Euclidean norm

```python

from thesis.oit import norm
baseline = norm(time_series)

```

#### Literature: SVD-based OIT

```python

from thesis.oit import svd
literature = svd(time_series)

```

#### Unique SVD using skewness

```python

from thesis.oit import usvd_skew
unique_skew = usvd_skew(time_series)

```

#### Unique SVD using mean

```python

from thesis.oit import usvd_mean
unique_mean = usvd_mean(time_series)

```

#### Unique SVD using absolute values

```python

from thesis.oit import usvd_abs
unique_absolute = usvd_abs(time_series)

```

#### Windowed SVD

```python

from thesis.oit import wsvd
windowed = wsvd(time_series)

```

#### Windowed Unique SVD using mean

(Techniques using `skewness` and `absolute` are performed similarly.)

```python

from thesis.oit import wusvd_mean
windowed_unique_mean = wusvd_mean(time_series)

```

### Detection

#### MTMM-DTW

```python

from pathlib import Path
from detection import mtmmdtw
Path('save_at_location').mkdir(parents=True, exist_ok=True)
mtmmdtw('save_at_location/name.npy', time_series, templates) 

```

#### VLMTMM-DTW using mean

```python

from pathlib import Path
from detection import novel_mtmmdtw_mean
Path('save_at_location').mkdir(parents=True, exist_ok=True)
novel_mtmmdtw_mean('save_at_location/name.npy', [[templates, time_series]]) 

```
