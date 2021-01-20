from pathlib import Path
import os.path
import pandas as pd
from numpy import nan
from typing import Sequence
from dataclasses import dataclass

data_folder = Path(__file__).parent.joinpath('dataset')
dataset_name = "household_power_consumption"
features: Sequence[str] = ("wind_speed",)

@dataclass
class ReadData:
    #dataset_name: str

    def load_data(self):
        filename_csv = data_folder.joinpath(f"{dataset_name}.csv")
        if os.path.isfile(filename_csv):
            print("Reading: ", filename_csv)
            dataset = pd.read_csv(filename_csv, header=0, infer_datetime_format=True, 
                               parse_dates=['datetime'], index_col=['datetime'])
            # summarize
            print(dataset.shape)
        else:
            filename_txt = data_folder.joinpath(f"{dataset_name}.txt")
            print("Reading and pre-processing: ", filename_txt)
            dataset = pd.read_csv(filename_txt, sep=';', header=0, low_memory=False, 
                               infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
            # mark all missing values
            dataset.replace('?', nan, inplace=True)
            print('Missing values:', dataset.isnull().sum()) 
            
            # add a column for for the remainder of sub metering
            values = dataset.values.astype('float32')
            dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
            # save updated dataset
            dataset.to_csv(filename_csv)
            print("Saving: ", filename_csv)
            # summarize
            print(dataset.shape)
        dataset.values.astype(float)
        return dataset
    
    def resample(self, dataset, resampling_rate: str = "D"):
        filename_res_csv = data_folder.joinpath(f"{dataset_name}_resampled_{resampling_rate}.csv")
        if os.path.isfile(filename_res_csv):
            dataset_res = pd.read_csv(filename_res_csv, header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
            return dataset_res
        else:
            # resample minute data to total for each day
            # resample data to daily
            daily_groups = dataset.resample(resampling_rate)
            daily_data = daily_groups.sum()
            # summarize
            print(daily_data.shape)
            # save
            #daily_data = daily_data.drop(daily_data["2006-12-16"].index)
            #daily_data = daily_data.drop(daily_data["2010-11-26"].index)
            #daily_data
            daily_data.to_csv(filename_res_csv)
            return daily_groups;

