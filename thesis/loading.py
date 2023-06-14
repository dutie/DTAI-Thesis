"""
thesis.loading
~~~~~~~~~~~~~~~~~

Loading the data of the dataset "Yurtman,Aras & Barshan,Billur. (2022). Physical Therapy Exercises Dataset. UCI Machine Learning Repository."

:author: Mats Fockaert
:copyright: Copyright 2023 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np
import os
import matplotlib.pyplot as plt

test_type      = list[list[float]]
templates_type = list[list[list[float]]]
sensor_type    = list[templates_type, test_type]
exercise_type  = list[sensor_type]
subject_type   = list[exercise_type]

class Loading:
    def __init__(self, home_folder):
        self.home_folder = home_folder
        self.annotated   = []
        self.time_series = np.empty((5, 8, 5,2), dtype=object)
    
    def get_time_stamps(self, path):
        times = np.genfromtxt(path, dtype=list, delimiter=',', names=True)
        return times

    def split_template_data(self, time_stamps, data_):
        data = np.array(data_)
        temp1_time, temp2_time, temp3_time = time_stamps[0, :], time_stamps[1, :], time_stamps[2, :]
        temp1 = data[round(temp1_time[0]):round(temp1_time[1]), :]
        temp2 = data[round(temp2_time[0]):round(temp2_time[1]), :]
        temp3 = data[round(temp3_time[0]):round(temp3_time[1]), :]
        return [temp1, temp2, temp3]

    def load_all(self):
        """
        Sets `self.time_series` to have a list of subjects that each have a list of exercises that each have a list of sensors that each have their templates and test time series.

        `self.time_series: list[subject_type]`.
        """
        template_times = None
        for dirpath, dirnames, filenames in os.walk(self.home_folder, topdown=True):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if filename == 'annotations.csv':
                    su_id, ex_id = list(map(lambda str_val: int(str_val[1]), dirpath.split('\\')[4:]))
                    self.annotated.append({"id":f"s{su_id}:e{ex_id}", "data":self.get_time_stamps(file_path)})
                elif filename == 'template_times.txt':
                    template_times = np.genfromtxt(file_path, delimiter=';', skip_header=1, usecols=(1, 2))
                elif filename in ['test.txt', 'template_session.txt']:
                    su_id, ex_id, se_id = list(map(lambda str_val: int(str_val[1]), dirpath.split('\\')[4:]))
                    if filename == 'test.txt':
                        data = np.genfromtxt(file_path, delimiter=';', skip_header=1, usecols=(1, 2, 3))
                        self.time_series[su_id-1,ex_id-1,se_id-1, 1] = data
                    elif filename == 'template_session.txt':
                        data = np.genfromtxt(file_path, delimiter=';', skip_header=1, usecols=(1, 2, 3))
                        self.time_series[su_id-1,ex_id-1,se_id-1, 0] = self.split_template_data(template_times, data)
                        
    def get_sensors_for_exercises_for_subjects(self, sensors: list[float], exercises: list[float], subjects: list[float] ) -> list[subject_type]:
        """
        Note all from 1 - end, not from 0.
        """
        new_subjects = []
        for subject in subjects:
            new_exercises = []
            for exercise in exercises:
                new_sensors = []
                for sensor in sensors:
                    new_sensor = self.time_series[subject-1, exercise-1, sensor-1]
                    new_sensors.append(new_sensor)
                new_exercises.append(new_sensors)
            new_subjects.append(new_exercises)
        return np.array(new_subjects, dtype=object)
    
    def get_sensor(self, subject: list[int], exercise: list[int], sensor: list[int]) -> sensor_type:
        """
        Note all from 1 - end, not from 0.
        """
        return self.time_series[subject-1, exercise-1, sensor-1]
    
    def get_exercise(self, subject: list[int], exercise: list[int]) -> exercise_type:
        """
        Note all from 1 - end, not from 0.
        """
        return self.time_series[subject-1, exercise-1]
    
    def get_subject(self, subject: list[int]) -> subject_type: 
        """
        Note all from 1 - end, not from 0.
        """
        return self.time_series[subject-1]
            
    def get_relevant_data_only(self):
        """
        Each exercise has one main sensor that will be most prevelant to the exercise. This sensor is selected and returned.

        returns List[List[List[templates_type, test_type]]]
        """
        updated_subjects = []
        exercises = np.arange(8)
        sensors   = [1,3,1,1,1,1,1,1]
        for subject in self.time_series:
            tdata = subject[exercises]
            data  = []
            for ide, exercise in enumerate(tdata):
                sensor = exercise[sensors[ide]]
                data.append([sensor])
            updated_subjects.append(data)
        return updated_subjects

        