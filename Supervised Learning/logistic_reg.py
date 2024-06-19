import pandas as pd
import numpy as np
import logging
import math


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%y %I:%M:%S%p ')


class ClassificationModels:
    def sigmoid(self, probablities):
        maximum = max(probablities)
        return maximum
        
    
    def logistic_reg(self, X_train, Y_train, X_test, Y_tests):
        # B(slope) -> inverse((X transpose) * X) * (X transpose) * Y
        X_transpose = X_train.to_numpy().transpose()
        X_trans_X = X_transpose @ X.to_numpy()
        X_trans_X_inv = np.linalg.pinv(X_trans_X)
        X_trans_Y = X_transpose @ Y
        B = X_trans_X_inv @ X_trans_Y
        logging.info(f"B(slope) -> {B}")

        # print(X_train.to_numpy()[0])

        # linear model y = mx + b
        z = [sum(X_train.to_numpy()[i] * B) for i in range(len(X_train.to_numpy()))]
        logging.info(f"z(sum of all mx+b) -> {z}")
        
        # e* z inverse
        e_zinv = [math.exp(z_val) for z_val in z]
        logging.info(f"e inverse -> {e_zinv}")

        # sigmoid function
        probablities = [(1 / (1 + e)) for e in e_zinv]
        logging.info(f"Output probablities -> {probablities}")

        # return probablities


if __name__ == "__main__":
    # train
    X = pd.DataFrame(data={
            'a':[1, 2, 3, 4, 5, 6, 7],
            'b':[8, 9, 10, 11, 12, 13, 14],
            'c':[15, 16, 17, 18, 19, 20, 21],
            'd':[22, 23, 24, 25, 26,27, 28]
            })
    Y = [1, 0, 0, 1, 1, 0, 1]

    # test 
    X_test = pd.DataFrame(data={'a':[28],
            'b':[29],
            'c':[30],
            'd':[31]})
    
    Y_test= [0]

    classification_models = ClassificationModels()
    variables = classification_models.logistic_reg(X, Y, X_test, Y_test)