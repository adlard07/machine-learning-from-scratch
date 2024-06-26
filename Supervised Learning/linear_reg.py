import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

class LinearRegression:
    def linear_regression(self, X, Y):
        try:
            n = len(Y)
            # -------------multivariet regression-------------
            if np.array(X).shape[-1] > 1:
                try:
                    if type(Y) == "<class 'pandas.core.frame.DataFrame'>":
                        Y = Y.to_numpy()
                    else:
                        Y = np.array(Y)
                    
                    # B(slope) formula -> inverse((X transpose) * X) * (X transpose) * Y
                    X_transpose = X.to_numpy().transpose()
                    X_trans_X = np.array(X_transpose @ X.to_numpy())
                    X_trans_X_inv = np.linalg.pinv(X_trans_X)
                    X_trans_Y = X_transpose @ Y
                    B = X_trans_X_inv @ X_trans_Y
                    logging.info(f"The coefficients X1, X2,... Xn -> {B}")
                    
                    y_pred = X.to_numpy() @ B
                    logging.info(f"Predictions -> {y_pred}")

                    residuals = Y - y_pred
                    MSE = np.mean(residuals**2)
                    logging.info(f"Mean Squared Error -> {MSE}")
                    
                    return y_pred, B, B[0], MSE 
                except Exception as e:
                    logging.error('Exception occured at', e)


            # -------------simple linear regression--------------
            else:
                try:
                    X = X[0]
                    sum_of_X = sum(np.array(X))
                    sum_of_Y = sum(np.array(Y))
                    XY, X_sqr = [], []
                    for i in range(len(X)):
                        xy = X[i]*Y[i]
                        x_sqr = X[i]*X[i]

                        XY.append(xy)
                        X_sqr.append(x_sqr)
                    sum_of_XY = sum(XY)
                    sum_of_X_sqr = sum(X_sqr)

                except Exception as e:
                    logging.error(f'The lengths of {X} and {Y} do not match')
                    
                B = ((n * sum_of_XY) - (sum_of_X*sum_of_Y)) / ((n * sum_of_X_sqr) - (sum_of_X * sum_of_X))
                logging.info(f"Slope -> {B}")

                intercept = ((sum_of_Y - (B * sum_of_X)) / (n))
                logging.info(f"Intercept(coefficient) -> {intercept}")

                y_pred, error = [], []
                for x in X:
                    pred = intercept + (B * x)
                    err = pred - x
                    y_pred.append(pred)
                    error.append(err)
                logging.info(f"Predictions -> {y_pred}")
                    
                MSE = [np.mean(err**2) for err in error]
                logging.info(f"Mean Squared Error: {MSE}")

                return y_pred, B, intercept, MSE 

        except Exception as e:
            logging.error('An exception occured at: ', e)



if __name__ == '__main__':
    # X = pd.DataFrame(data=[1, 2, 3, 4, 5, 6, 7])

    X = pd.DataFrame(data={
        'a':[1, 2, 3, 4, 5, 6, 7],
        'b':[8, 9, 10, 11, 12, 13, 14],
        'c':[15, 16, 17, 18, 19, 20, 21],
        'd':[22, 23, 24, 25, 26,27, 28]
        })

    Y = [1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16]

    x_test = pd.DataFrame(data=[2, 9, 16, 23])
    y_test = [1.4, 8, 12, 26]

    linear_reg = LinearRegression()
    variables = linear_reg.linear_regression(X, Y)