# Main purpose of this data tranformation is to do the feature engineering, data cleaning 
# Convert the categorical features into numerical features 

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# This config class will give any path that will be 
# requiring any inputs it may probably require for my data and transformation component
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # All the pickle files which will be responsible for converting the categorical features into numerical
    # if you want to perform a standard scalar and etc.
    def get_data_transformer_object(self):
        '''
        This function responsiblie for data transformation
        '''
        try:
            # Numerical features
            numerical_columns = ["writing_score", "reading_score"]
            
            # Categorical features
            categorical_columns = [
                "gender", "race_ethnicity",
                "parental_level_of_education", "lunch",
                "test_preparation_course"
            ]

            # Create a numerical pipeline and handling the missing value. 
            # This pipeline should run on training dataset
            num_pipeline = Pipeline(
                # Handling the missing values and doing the standard scaling
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
            )

            # Create a categorical pipeline and handling the missing value.
            # Converting categorical to numerical value using OneHotEncoding 
            # This pipeline should run on training dataset
            ''' StandardScaler(with_mean=False)
                The reason that adding with_mean=False resolved my error is that the StandardScaler is subtracting the mean from each feature, which can result in some features having negative values. However, StandardScaler() alone assumes that the features have positive values, which can cause issues when working with features that have negative values.
                By setting with_mean=False, the StandardScaler does not subtract the mean from each feature, and instead scales the features based on their variance. This can help preserve the positive values of the features and avoid issues.
            '''
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # logging.info("Numerical colums standard scaling completed")
            # logging.info("Categorical colums encoding completed")

            logging.info(f"Categorical colums: {categorical_columns}")
            logging.info(f"Numerical colums: {numerical_columns}")

            # Combining the numerical and categorical pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    # Initiating the data transformation inside this function    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data are completed")

            logging.info("Obtaining preprocesing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            
            # Numerical features
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
        


        