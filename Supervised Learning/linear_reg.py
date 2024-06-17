import numpy 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class LinearModels:
    def simple_linear_regression(self, X: list, Y: list):
        try:
            try:
                sum_of_X = sum(X)
                sum_of_Y = sum(Y)
                n = len(X)
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

            slope = ((n * sum_of_XY) - (sum_of_X*sum_of_Y)) / ((n * sum_of_X_sqr) - (sum_of_X * sum_of_X))
            intercept = ((sum_of_Y - (slope * sum_of_X)) / (n))

            y_pred, error = [], []
            for x in X:
                pred = intercept + (slope * x)
                err = pred - x    
                err = err * err
                y_pred.append(pred)
                error.append(err)

            mse = sum(error) / n
        
        except Exception as e:
            logging.error('An exception occured at: ', e)

        return slope, intercept, y_pred, mse
    


if __name__ == '__main__':
    X = [1, 2, 3, 4, 5, 6, 7]
    Y = [1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16]

    linear_reg = LinearModels()
    slope, intercept, y_pred, mse = linear_reg.simple_linear_regression(X, Y)
    print(y_pred, mse)