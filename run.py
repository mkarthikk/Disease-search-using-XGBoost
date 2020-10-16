import os
import logging
import traceback
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class Data:
    def __init__(self):
        self.col_num = {}

    def parse_data(self, file, drop_cols=[]):
        count = 0
        data = pd.read_csv(file)
        columns = np.array(data.columns)
        print(columns)
        for col in drop_cols:
            if col in data.columns:
                X = data.drop([col], axis=1).copy()
        try:
            X = X.astype('float64').copy()
            y = data['prognosis'].copy()

            for disease in y:
                if disease not in self.col_num.keys():
                    self.col_num.update({disease: count})
                    count += 1
            y = data['prognosis'].map(self.col_num)
        except Exception as e:
            traceback.print_exc()
            logger.info(f"Prognosis should be a column. Also data should only contain int, float, boolean")
        return X, y, self.col_num


class XGBModel:

    def train_model(self):

        file = ""
        X, y, labels = Data().parse_data(file, ["prognosis"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=54, test_size=0.25)
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)
        params = {
            'max_depth': 4,
            'eta': 0.3,
            'objective': 'multi:softprob',
            'num_class': 41
        }
        epochs = 10
        model = xgb.train(params, train, epochs)
        predictions = model.predict(test)
        model.save_model("DiseaseSearch.json")
        pred_arr = []
        for index, value in enumerate(predictions):
            pred_arr.append(np.argmax(value))
        pred_np_arr = np.array(pred_arr)
        a_score = accuracy_score(pred_np_arr, y_test)
        print(labels)
        print(a_score)
        return model

def query_prediction():
    xgbmodel = XGBModel()
    model = xgbmodel.train_model()
    return model

query_prediction()