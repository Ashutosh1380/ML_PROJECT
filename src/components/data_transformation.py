import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
## Column Transformer allows us to apply different preproccesing steps to different columns(features) within one pipeline
from sklearn.compose import ColumnTransformer
## SimpleImputer is used for handling missing values
from sklearn.impute import SimpleImputer
## Helps you chain multiple data processing steps and a final estimator (model) into one object.
from sklearn.pipeline import Pipeline
## Onehotencoder for categorical features and standardscaler for numerical features
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


## similar to the dataingestionconfig, is used so that if we require any inputs, they are specified here
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
 
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )


            ## after applying one hot encoding, for eg some column has big values after one hot encoding
            ## to tackle that problem we apply standard scaler after applying one hot encoding

            ## NOT RELATED TO ALL OF THIS, BUT FIT_TRANSFORM MAI FIT LEARNS THE PATTERNS FROM THE DATA AND TRANSFORM IS USED TO APPLY
            ## APPLY THE LEARNED PARAMETERS/PATTERNS IN THE DATA, THEREFORE IN TRAIN DATASET WE DO FIT_TRANSFORM,
            ## AND TO SIMULATE THE SITUATION OF UNSEEN DATA WE DO ONLY TRANSFORM ON TEST DATA, SO THAT IT DOES NOT LEARN FROM THE DATA OF TEST
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ## name(you want to give), object name(like above), list_of_columns
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )
            ## we return the preprocessor meaning we return the preprocessing steps that we have done
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    ## train_path and test_path you are getting from data_ingestion.py
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            ## axis = 1 means drop along columns
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            ## now we apply the preprocessing steps from get_data_transformer_object to the input_features

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            ## doing concatenation
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            ## doing concatenation
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)