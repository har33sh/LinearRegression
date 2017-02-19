Comaprision (L2 gradient descent,L2 closed form solution)
L2 gradient descent gave results close to L2 closed from solution when the alpha (learning factor of the Data) was too less 1e-4 and threshold error for the converge the gradient function 1e6 or 1e7, however the processing duration was too high. When the error and the learning factor was reduced the gradient was able to converge faster resulting in a slight increase in RMSE based on alpha and error threshold.

Feature Enginering:
Feature Engineering gives better results compared taking the features directly. Adding features, ~root of the features given (power 0.3-0.6) and  squares (power 1.8-2.2) gave less RMSE when the predicted output.

Normalization on the data is done to make the gradient converge faster. Normalization is done after the feature engineering to get better quality of the data.


Observations:
I have taken 40% of the given data for training, and the rest of 60% data taken for cross validation. Limiting the data that has been taken helps to reduce overfitting of the model. When the weights were calculated on the entire given data, RMSE of the predicted output and the given output was very less, but the RMSE of the predicted ouput based on the test data on kaggle showed a significant difference, which is likely due to the overfitting.


Results: (RMSE based on kaggle submissions)
L2 Norm = 3.56048
L1.2 Norm = 3.56176
L1.3 Norm = 3.35686
L1.4 Norm = 3.56235
