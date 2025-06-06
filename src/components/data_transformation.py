# import sys
# import os # to interact with computer system to create or access folders,directories,files
# import numpy as np
# import pandas as pd

# from pymongo import MongoClient
# from src.constant import *
# from src.exception import CustomException
# from src.logger import logging
# from src.data_access import phising_data
# from src.utils.main_utils import MainUtils
# from dataclasses import dataclass

# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer # for missing value treatment
# from sklearn.preprocessing import StandardScaler,OneHotEncoder #for scaling and data encoding()
# from imblearn.over_sampling import RandomOverSampler # to treat data imbalance
# @dataclass #decorator>>This provide you boiler plate of the code so you dont need to write same things or you can directly assign names(eg) withourt using self
# class DataTransformationconfig:
#     data_transformation_dir=os.path.join(artifact_folder,"data_transformation")
#     transformed_train_file_path=os.path.join(data_transformation_dir,"train.npy")
#     transformed_test_file_path=os.path.join(data_transformation_dir,"test.npy")
#     transformed_object_file_path=os.path.join(data_transformation_dir,"preprocessing.pkl")


# class DataTransformation:

#     def __init__(self,valid_data_dir):
#         self.valid_data_dir=valid_data_dir
#         self.data_transformation_config=DataTransformationconfig() # Creating an object of the DataTransformationConfig class to access its variables and configurations.
#         self.utils=MainUtils()
    

#     @staticmethod ##by this you just call the below function with the class name DataTransformation.get_merged_batch_data()
#     def get_merged_batch_data(valid_data_dir:str) ->pd.DataFrame:
#         """
#         Method Name :   get_merged_batch_data
#         Description :   This method reads all the validated raw data from the valid_data_dir and returns a pandas DataFrame containing the merged data. 
        
#         Output      :   a pandas DataFrame containing the merged data 
#         On Failure  :   Write an exception log and then raise an exception
        
#         Version     :   1.2
#         Revisions   :   moved setup to cloud
#         """
#         ''' Polished Explanation:
#             First, os.listdir(valid_data_dir) retrieves all file names in the directory containing valid CSVs. 
#             Then, for each file, we read it into a pandas DataFrame and append it to a list (csv_data). 
#             Finally, we use pd.concat() to combine all these individual DataFrames into one merged dataset.'''
        

#         try:
#             raw_files=os.listdir(valid_data_dir)
#             csv_data=[]
#             for filename in raw_files:
#                 data=pd.read_csv(os.path.join(valid_data_dir,filename))
#                 csv_data.append(data)
#             merged_data=pd.concat(csv_data)

#             return merged_data
#         except Exception as e:
#             raise CustomException(e,sys)


#     def initiate_data_transformation(self):
#          """
#             Method Name :   initiate_data_transformation
#             Description :   This method initiates the data transformation component for the pipeline 
            
#             Output      :   data transformation artifact is created and returned 
#             On Failure  :   Write an exception log and then raise an exception
            
#             Version     :   1.2
#             Revision"""
#          logging .info("Entered initiate_data_transformation method of DataTransformation class")


#          try:
#              #access the dataframe
#              dataframe=self.get_merged_batch_data(valid_data_dir=self.valid_data_dir)
#              dataframe=self.utils.remove_unwanted_spaces(dataframe)
#              dataframe.replace('?',np.nan,inplace=True)  # replacing '?' with NaN values for imputation

#              x=dataframe.drop(TARGET_COLUMN,axis=1)
#              y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)  # replacing the values of the target column

#              sampler = RandomOverSampler() # we make sure that data gets balanced or we have treated the imbalance data
#              x_sampled,y_sampled=sampler.fit_resample(x,y)

#              x_train,x_test,y_train,y_test=train_test_split(x_sampled,y_sampled,test_size=0.2)
#              y_train=y_train.ravel()
#              y_test=y_test.ravel()

#              preprocessor=SimpleImputer(strategy="most_frequent")#mode
#              x_train_scaled=preprocessor.fit_transform(x_train)
#              x_test_scaled=preprocessor.transform(x_test) #we only do transformation for test data

#              preprocessor_path=self.data_transformation_config.transformed_object_file_path
#              os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
#              self.utils.save_object(file_path=preprocessor_path,
#                                    obj=preprocessor)
             
#              return x_train_scaled,x_test_scaled,y_train,y_test,preprocessor_path
         
#          except Exception as e:
#              raise CustomException(e,sys) from e
             
import sys
import os
import numpy as np
import pandas as pd

from pymongo import MongoClient
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.data_access import phising_data
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

@dataclass
class DataTransformationconfig:
    data_transformation_dir=os.path.join(artifact_folder,"data_transformation")
    transformed_train_file_path=os.path.join(data_transformation_dir,"train.npy")
    transformed_test_file_path=os.path.join(data_transformation_dir,"test.npy")
    transformed_object_file_path=os.path.join(data_transformation_dir,"preprocessing.pkl")


class DataTransformation:

    def __init__(self,valid_data_dir):
        self.valid_data_dir=valid_data_dir
        self.data_transformation_config=DataTransformationconfig()
        self.utils=MainUtils()
    
    @staticmethod
    def get_merged_batch_data(valid_data_dir:str) ->pd.DataFrame:
        try:
            raw_files=os.listdir(valid_data_dir)
            csv_data=[]
            for filename in raw_files:
                data=pd.read_csv(os.path.join(valid_data_dir,filename))
                csv_data.append(data)
            merged_data=pd.concat(csv_data)

            return merged_data
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self):
         logging .info("Entered initiate_data_transformation method of DataTransformation class")

         try:
             dataframe=self.get_merged_batch_data(valid_data_dir=self.valid_data_dir)
             dataframe=self.utils.remove_unwanted_spaces(dataframe)
             dataframe.replace('?',np.nan,inplace=True)

             x=dataframe.drop(TARGET_COLUMN,axis=1)
             y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1) # y is now a 1D numpy array or (N,1) 2D array

             sampler = RandomOverSampler() # Added random_state for reproducibility
             x_sampled,y_sampled=sampler.fit_resample(x,y)

             # x_sampled and y_sampled are guaranteed to be aligned and have the same number of samples
             # Debugging: Print shapes after oversampling
             print(f"DEBUG(DT): Shapes after oversampling: x_sampled.shape={x_sampled.shape}, y_sampled.shape={y_sampled.shape}")

             x_train,x_test,y_train,y_test=train_test_split(x_sampled,y_sampled,test_size=0.2, random_state=1) # Added random_state
             
             # Debugging: Print shapes after train_test_split
             print(f"DEBUG(DT): Shapes after train_test_split: x_train.shape={x_train.shape}, y_train.shape={y_train.shape}, x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")

             # Removed y_train.ravel() and y_test.ravel() here. ModelTrainer will handle flattening.
             # This avoids any potential for '.values' being incorrectly applied if y was already a NumPy array.

             preprocessor=SimpleImputer(strategy="most_frequent")
             x_train_scaled=preprocessor.fit_transform(x_train)
             x_test_scaled=preprocessor.transform(x_test)
             
             # Debugging: Print shapes after imputation
             print(f"DEBUG(DT): Shapes after imputation: x_train_scaled.shape={x_train_scaled.shape}, x_test_scaled.shape={x_test_scaled.shape}")

             preprocessor_path=self.data_transformation_config.transformed_object_file_path
             os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
             self.utils.save_object(file_path=preprocessor_path,
                                   obj=preprocessor)
             
             return x_train_scaled,y_train,x_test_scaled,y_test,preprocessor_path
         
         except Exception as e:
             raise CustomException(e,sys) from e


