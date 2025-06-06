# import sys
# from typing import Generator,List,Tuple,Dict
# import os
# import pandas as pd
# import numpy as np

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.compose import ColumnTransformer
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# from src.constant import *
# from src.exception import CustomException
# from src.logger import logging
# from src.utils.main_utils import MainUtils

# from dataclasses import dataclass

# @dataclass
# class ModelTrainerConfig:
#     model_trainer_dir=os.path.join(artifact_folder,"model_trainer")
#     trained_model_path=os.path.join(model_trainer_dir,"trained_model","model.pkl")
#     expected_accuracy=0.45
#     model_config_file_path=os.path.join("config","model.yaml")


# class VisibilityModel:
#     def __init__(self,preprocessing_object:ColumnTransformer,trained_model_object):

#         self.preprocessing_object=preprocessing_object # Applies preprocessing transformations to all designated columns

#         self.trained_model_object=trained_model_object
        
#     def predict(self,X:pd.DataFrame)->pd.DataFrame:
#         logging.info("Entered predict class of Modeltraining class")

#         try:
#             logging.info("Using the trained model to get the prediction")

#             transformed_feature=self.preprocessing_object.transform(X)

#             logging.info("Used the training model to get predictions")
#             return self.trained_model_object.predict(transformed_feature)

#         except Exception as e:
#             raise CustomException(e,sys) from e
        

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"
    
#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"
    

# class ModelTrainer:
#     def __init__(self):

#         self.model_trainer_config=ModelTrainerConfig()

#         self.utils=MainUtils()

#         self.models={
#                 "GaussianNB":GaussianNB(),
#                 "XGBClassifier":XGBClassifier(objective="binary:logistic"),
#                 "LogisticRegression":LogisticRegression()

#         }

#     # def evaluate_models(self, X_train,
#     #                 X_test,
#     #                 y_train,
#     #                  y_test,
#     #                  models:Dict[str,object]) ->dict:
        
#     #     try:
#     #         #Ensure y is 1d
#     #         # Flatten target arrays
#     #         if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
             
                
             
#     #          y_train = y_train.values.ravel()
#     #         else:      
#     #          y_train = y_train.ravel()

#     #         if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):

#     #             y_test = y_test.values.ravel()
#     #         else:
                 
#     #              y_test = y_test.ravel()

#     #         report={}

#     #         for model_name,model in models.items():
                
#     #             model.fit(X_train,y_train)

#     #             y_train_pred=model.predict(X_train)

#     #             y_test_pred=model.predict(X_test)

#     #             train_model_score=accuracy_score(y_train,y_train_pred)

#     #             test_model_score=accuracy_score(y_test,y_test_pred)

#     #             report[model_name]={"Train accuracy":train_model_score,"Test_model_accuracy":test_model_score}
#     #             print("evaluation Report",report)

#     #         return report
#     #     except Exception as e:
#     #         raise CustomException(e,sys)
        
    
#     def evaluate_models(self, X_train,
#                     X_test,
#                     y_train,
#                      y_test,
#                      models:Dict[str,object]) ->dict:
        
#         try:
#             #Ensure y is 1d
#             # Flatten target arrays
#             if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
#                 y_train = y_train.values.ravel()
#             else:      
#                 # y_train is already a numpy array, no need for .values
#                 y_train = y_train.ravel()

#             if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
#                 y_test = y_test.values.ravel()
#             else:
#                 # y_test is already a numpy array, no need for .values
#                 y_test = y_test.ravel()

#             report={}

#             for model_name,model in models.items():
                
#                 model.fit(X_train,y_train)

#                 y_train_pred=model.predict(X_train)

#                 y_test_pred=model.predict(X_test)

#                 train_model_score=accuracy_score(y_train,y_train_pred)

#                 test_model_score=accuracy_score(y_test,y_test_pred)

#                 report[model_name]={"Train accuracy":train_model_score,"Test_model_accuracy":test_model_score}
#                 print("evaluation Report",report)

#             return report
#         except Exception as e:
#             raise CustomException(e,sys)
        


#     def get_best_model(self,x_train,y_train,x_test,y_test):

#         try:

#             model_report:dict=self.evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=self.models)

#             print(model_report)
#             best_model_score = max([v["Test_model_accuarcy"] for v in model_report.values()])
#             best_model_name = [k for k, v in model_report.items() if v["Test_model_accuarcy"] == best_model_score][0]

#             best_model_object=self.models[best_model_name] #you are storing the best model

#             return best_model_name,best_model_object,best_model_score
#         except Exception as e:
#             raise CustomException(e,sys)
            
        
#     def finetune_best_model(self,best_model_object:object,best_model_name,X_train,y_train) ->object:

#         try:
#                 # first you will go to the model configuartion then  go the model config file path which lead you to model.yaml where you have ,odel selection inside model selection you have many models but you will select the best model and take its param_grid
#             model_param_grid=self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

#             grid_search=GridSearchCV(best_model_object,param_grid=model_param_grid,cv=5,n_jobs=-1,verbose=1)

#             grid_search.fit(X_train,y_train)

#             best_params=grid_search.best_params_

#             print("best params are:",best_params)

#             finetuned_model=best_model_object.set_params(**best_params)

#             return finetuned_model
            
#         except Exception as e:
#             raise CustomException(e,sys)
            

#     def initiate_model_trainer(self,
#                                x_train,
#                                y_train,
#                                x_test,
#                                y_test,
#                                preprocessor_path):
#          try:

#             logging.info(f"Splitting training and testing input and target feature")

#             logging.info(f"Extracting model config file path")
#             #you are extractiog the trained model from aws
#             preprocessor = self.utils.load_object(file_path=preprocessor_path)
#             if len(y_train.shape)>1:
#                 y_train=y_train.ravel()
#             if len(y_test.shape)>1:
#                 y_test=y_test.ravel()
#             logging.info(f"Extracting model config file path")

#             model_report: dict = self.evaluate_models(
#                 X_train=x_train,
#                 y_train=y_train,
#                 X_test=x_test,
#                 y_test=y_test, models=self.models)

#             ## To get best model score from dict
#             best_model_score = max([v["Test_model_accuarcy"] for v in model_report.values()])
#             best_model_name = [k for k, v in model_report.items() if v["Test_model_accuarcy"] == best_model_score][0]


#             ## To get best model name from dict


#             best_model = self.models[best_model_name]

#             best_model = self.finetune_best_model(
#             best_model_name=best_model_name,
#             best_model_object=best_model,
#             X_train=x_train,
#             y_train=y_train
#             )

#             best_model.fit(x_train, y_train)
#             y_pred = best_model.predict(x_test)
#             best_model_score = accuracy_score(y_test, y_pred)

#             print(f"best model name {best_model_name} and score: {best_model_score}")

#             if best_model_score < 0.5:
                 
#                 raise Exception("No best model found with an accuracy greater than the threshold 0.5")

#             logging.info(f"Best found model on both training and testing dataset")

#             custom_model = VisibilityModel(
#             preprocessing_object=preprocessor,
#             trained_model_object=best_model
#                 )

#             logging.info(
#                 f"Saving model at path: {self.model_trainer_config.trained_model_path}"
#                 )

#             os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

#             self.utils.save_object(
#                 file_path=self.model_trainer_config.trained_model_path,
#                 obj=custom_model,
#                 )

#             self.utils.upload_file(from_filename=self.model_trainer_config.trained_model_path,
#                                    to_filename="model.pkl",
#                                    bucket_name=AWS_S3_BUCKET_NAME)

#             return best_model_score

#          except Exception as e:
#             raise CustomException(e, sys)


import sys
from typing import Generator,List,Tuple,Dict
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_dir=os.path.join(artifact_folder,"model_trainer")
    trained_model_path=os.path.join(model_trainer_dir,"trained_model","model.pkl")
    expected_accuracy=0.45
    model_config_file_path=os.path.join("config","model.yaml")


class VisibilityModel:
    def __init__(self,preprocessing_object:ColumnTransformer,trained_model_object):

        self.preprocessing_object=preprocessing_object

        self.trained_model_object=trained_model_object
        
    def predict(self,X:pd.DataFrame)->pd.DataFrame:
        logging.info("Entered predict class of Modeltraining class")

        try:
            logging.info("Using the trained model to get the prediction")

            transformed_feature=self.preprocessing_object.transform(X)

            logging.info("Used the training model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CustomException(e,sys) from e
        

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    

class ModelTrainer:
    def __init__(self):

        self.model_trainer_config=ModelTrainerConfig()

        self.utils=MainUtils()

        self.models={
                "GaussianNB":GaussianNB(),
                "XGBClassifier":XGBClassifier(objective="binary:logistic"),
                "LogisticRegression":LogisticRegression()

        }
    def evaluate_models(self,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        models):
        try:

            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(X_train, y_train)  # Train model

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)

                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    
    def get_best_model(self,
                       x_train: np.array,
                       y_train: np.array,
                       x_test: np.array,
                       y_test: np.array):
        try:

            model_report: dict = self.evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=self.models
            )

            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score


        except Exception as e:
            raise CustomException(e, sys)

    
            
        
    def finetune_best_model(self,best_model_object:object,best_model_name,x_train,y_train) ->object:

        try:
            model_param_grid=self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search=GridSearchCV(best_model_object,param_grid=model_param_grid,cv=5,n_jobs=-1,verbose=1)

            grid_search.fit(x_train,y_train)

            best_params=grid_search.best_params_

            print("best params are:",best_params)
            finetuned_model=grid_search.best_estimator_


            return finetuned_model
            
        except Exception as e:
            raise CustomException(e,sys)
            

    def initiate_model_trainer(self,
                               x_train,
                               y_train,
                               x_test,
                               y_test,
                               preprocessor_path):
         try:

            logging.info(f"Splitting training and testing input and target feature")

            logging.info(f"Extracting model config file path")
            preprocessor = self.utils.load_object(file_path=preprocessor_path)
            
            # Correction: Removed .values as y_train/y_test are already numpy arrays
            # This handles cases where y_train/y_test might be 2D numpy arrays like (N, 1).
            
            logging.info(f"Extracting model config file path")

            model_report: dict = self.evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test, models=self.models)

            ## To get best model score from dict
            # Corrected typo: "Test_model_accuarcy" to "Test_model_accuracy"
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = self.models[best_model_name]

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                x_train=x_train,
                y_train=y_train
            )
            print(f"DEBUG(MT_init): Before final best_model fit: x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
            
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f"best model name {best_model_name} and score: {best_model_score}")

            if best_model_score < 0.5:
                 
                raise Exception("No best model found with an accuracy greater than the threshold 0.5")

            logging.info(f"Best found model on both training and testing dataset")

            custom_model = VisibilityModel(
            preprocessing_object=preprocessor,
            trained_model_object=best_model
                )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
                )

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=custom_model,
                )

            self.utils.upload_file(from_filename=self.model_trainer_config.trained_model_path,
                                   to_filename="model.pkl",
                                   bucket_name=AWS_S3_BUCKET_NAME)

            return best_model_score

         except Exception as e:
            raise CustomException(e, sys)
