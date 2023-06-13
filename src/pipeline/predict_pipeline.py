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
        radius_mean: float,
        texture_mean: float,
        smoothness_mean: float,
        compactness_mean: float,
        concavity_mean: float,
        symmetry_mean: float,
        fractal_dimension_mean:float,
        radius_se:float,
        texture_se:float,
        smoothness_se:float,
        compactness_se:float,
        concavity_se:float,
        concave_points_se:float,
        symmetry_se:float,
        fractal_dimension_se:float,
        smoothness_worst:float,
        compactness_worst:float,
        symmetry_worst:float,
        fractal_dimension_worst: float):

        self.radius_mean = radius_mean
        self.texture_mean = texture_mean
        self.smoothness_mean = smoothness_mean
        self.compactness_mean = compactness_mean
        self.concavity_mean = concavity_mean
        self.symmetry_mean = symmetry_mean
        self.fractal_dimension_mean = fractal_dimension_mean
        self.radius_se = radius_se
        self.texture_se = texture_se
        self.smoothness_se = smoothness_se
        self.compactness_se = compactness_se
        self.concavity_se = concavity_se
        self.concave_points_se = concave_points_se
        self.symmetry_se = symmetry_se
        self.fractal_dimension_se = fractal_dimension_se 
        self.smoothness_worst = smoothness_worst
        self.compactness_worst = compactness_worst
        self.symmetry_worst = symmetry_worst
        self.fractal_dimension_worst = fractal_dimension_worst
        
         

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "radius_mean" :  [self.radius_mean],
                "texture_mean" :  [self.texture_mean],
                "smoothness_mean" :  [self.smoothness_mean],
                "compactness_mean" :  [self.compactness_mean],
                "concavity_mean" :  [self.concavity_mean],
                "symmetry_mean" :  [self.symmetry_mean],
                "fractal_dimension_mean" : [self.fractal_dimension_mean],
                "radius_se" : [self.radius_se],
                "texture_se" : [self.texture_se],
                "smoothness_se" : [self.smoothness_se],
                "compactness_se" : [self.compactness_se],
                "concavity_se" : [self.concavity_se],
                "concave_points_se" : [self.concave_points_se],
                "symmetry_se" : [self.symmetry_se],
                "fractal_dimension_se" : [self.fractal_dimension_se],
             "smoothness_worst" : [self.smoothness_worst],
                "compactness_worst" : [self.compactness_worst],
                "symmetry_worst" : [self.symmetry_worst],
                "fractal_dimension_worst" :  [self.fractal_dimension_worst]
        }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
