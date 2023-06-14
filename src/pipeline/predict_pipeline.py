import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))



import pandas as pd
from exception import CustomException
from utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,       
        age: int,
        gender: int,
        chest_pain: int,
        rest_bps: int,
        cholestrol: int,
        fasting_blood_sugar: int,
        rest_ecg: int,
        thalach:int,
        exer_angina:int,
        old_peak:float,
        slope:int,
        ca:int,
        thalassemia:int
        ):

        self.age = age
        self.gender = gender
        self.chest_pain = chest_pain
        self.rest_bps = rest_bps
        self.cholestrol = cholestrol
        self.fasting_blood_sugar = fasting_blood_sugar
        self.rest_ecg = rest_ecg
        self.thalach = thalach
        self.exer_angina = exer_angina
        self.old_peak = old_peak
        self.slope = slope
        self.ca = ca
        self.thalassemia = thalassemia
        
         
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age" :  [self.age],
                "gender" :  [self.gender],
                "chest_pain" :  [self.chest_pain],
                "rest_bps" :  [self.rest_bps],
                "cholestrol" :  [self.cholestrol],
                "fasting_blood_sugar" :  [self.fasting_blood_sugar],
                "rest_ecg" : [self.rest_ecg],
                "thalach" : [self.thalach],
                "exer_angina" : [self.exer_angina],
                "old_peak" : [self.old_peak],
                "slope" : [self.slope],
                "ca" : [self.ca],
                "thalassemia" : [self.thalassemia]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
