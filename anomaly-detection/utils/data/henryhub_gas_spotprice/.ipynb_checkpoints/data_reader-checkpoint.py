from pathlib import Path
import os.path
import pandas as pd
from numpy import nan
from typing import Sequence
from dataclasses import dataclass

data_folder = Path(__file__).parent.joinpath('dataset')
dataset_name = "Henry_Hub_Natural_Gas_Spot_Price"
#features: Sequence[str] = ("wind_speed",)

@dataclass
class ReadData:
    #dataset_name: str

    def load_data(self):
        filename_csv = data_folder.joinpath(f"{dataset_name}.csv")
        if os.path.isfile(filename_csv):
            print("Reading: ", filename_csv)
            dataset = pd.read_csv(filename_csv, header=0, infer_datetime_format=True, 
                               parse_dates=['Day'], index_col=['Day'])
            dataset = dataset.rename({'Henry Hub Natural Gas Spot Price Dollars per Million Btu': 'price'}, axis = 'columns')
            dataset = self.drop_missing(dataset)
            # summarize
            print(dataset.shape)
        dataset = dataset.sort_index()
        return dataset
    
    def resample(self, dataset, resampling_rate: str = "D"):
        filename_res_csv = data_folder.joinpath(f"{dataset_name}_resampled_{resampling_rate}.csv")
        if os.path.isfile(filename_res_csv):
            dataset = pd.read_csv(filename_res_csv, header=0, infer_datetime_format=True, parse_dates=['Day'], index_col=['Day'])
            dataset = self.drop_missing(daset)
            dataset = dataset.sort_index()
            return dataset
        else:
            # resample minute data to total for each day
            # resample data to daily
            daily_groups = dataset.resample(resampling_rate)
            daily_data = daily_groups.sum()
            daily_data = self.drop_missing(daily_data)
            # summarize
            print(daily_data.shape)
            # save
            #daily_data = daily_data.drop(daily_data["2006-12-16"].index)
            #daily_data = daily_data.drop(daily_data["2010-11-26"].index)
            #daily_data
            daily_data.to_csv(filename_res_csv)
            daily_groups = daily_groups.sort_index()
            return daily_groups;
   
    def drop_missing(self,dataset):
        # checking missing values
        dataset = dataset.dropna() 
        # dropping missing valies
        print('....Dropped Missing value row....')
        print('Rechecking Missing values:', dataset.isnull().sum()) 
        # checking missing values
        return dataset;
  