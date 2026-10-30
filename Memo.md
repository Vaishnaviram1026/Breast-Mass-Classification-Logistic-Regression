# Logistic Regression from Scores to Decisions

## 1. Introduction and Understanding Logistic Regression

In this project, I explored how logistic regression works and how it can be used to make classification decisions. I started by looking at how a model converts a feature value into a score, then uses the sigmoid function to calculate a probability. I also learned that the probability and the final classification are not the same thing because the classification depends on the selected threshold.

In Part 1, I worked through four examples to understand this relationship.

| Case | Intercept | Coefficient | x | Score (z) | Probability | Threshold | Prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.6 | 0.6457 | 0.50 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.6 | 0.6457 | 0.70 | 0 |
| C | -1.0 | -0.8 | 2.0 | -2.6 | 0.0691 | 0.50 | 0 |
| D | 1.0 | 0.0 | 2.0 | 1.0 | 0.7311 | 0.50 | 1 |

One thing I noticed was that Cases A and B have exactly the same probability, but their predictions are different because the thresholds are different. Case C showed me how changing the sign of a coefficient changes the direction of the relationship. In Case D, the coefficient is zero, so changing the feature value would not affect the score. These examples helped me understand why selecting a threshold is an important part of classification.

## 2. Dataset and Data Preparation

For this project, I used the Wisconsin Diagnostic Breast Cancer dataset available through `sklearn.datasets.load_breast_cancer()`. The dataset contains 569 observations. I selected two features for the analysis: mean radius and mean texture. I also recoded the target variable so that malignant cases were represented by 1 and benign cases by 0. The dataset contains 212 malignant cases and 357 benign cases.

I divided the data into training, validation, and test sets using stratified splitting and a random seed of 42. This helped maintain the class proportions across the three sets. I used the training set to fit the standard scaler and then applied that same scaler to the validation and test sets. I followed this process to avoid data leakage, since information from the validation and test sets should not influence the training process.

## 3. Building and Comparing the Models

I implemented logistic regression from scratch using NumPy and gradient descent. I initialized the coefficients at zero and used the gradient to update them over 20,000 epochs. The training loss decreased from 0.693147 to 0.246834. This showed that the model was reducing its prediction error during training.

The loss function used was binary cross-entropy. It measures how close the predicted probabilities are to the actual class labels. Minimizing cross-entropy is equivalent to maximizing the likelihood of the observed outcomes under the logistic regression model.

I then compared my NumPy implementation with scikit-learn's logistic regression model.

| Parameter | NumPy coefficient | scikit-learn coefficient |
|---|---:|---:|
| Intercept | -0.7271724042 | -0.7271724029 |
| Mean radius | 3.6567941718 | 3.6567941736 |
| Mean texture | 1.0098715332 | 1.0098715320 |

The two sets of coefficients were almost identical. The mean absolute difference between their validation probabilities was approximately 0.000000000119. Their validation log-loss values were also almost the same, at approximately 0.285809. This comparison helped me confirm that my NumPy implementation was producing results consistent with the scikit-learn model.

The mean radius coefficient was positive at approximately 3.657. Since the features were standardized, this means that a one-standard-deviation increase in mean radius is associated with an increase of approximately 3.657 in the predicted log-odds of malignancy, when mean texture is held constant. This describes the model's estimated relationship and should not be interpreted as a causal effect.

## 4. Model Evaluation and Baseline Comparison

I also compared the logistic regression model against a baseline model that predicts using the training-set class distribution. The baseline achieved test accuracy of 0.631579, but it predicted every test observation as benign. As a result, it missed all 42 malignant cases in the test set.

The NumPy logistic regression model using the default threshold of 0.50 performed much better. Its test accuracy was 0.868421, precision was 0.885714, recall was 0.738095, and F1 score was 0.805195. Its confusion matrix contained 68 true negatives, 4 false positives, 11 false negatives, and 31 true positives.

This comparison showed me why accuracy should not be considered on its own. Although the baseline had an accuracy above 60%, it failed to identify any malignant cases. Looking at recall, precision, F1, and the confusion matrix gave me a much clearer understanding of model performance.

## 5. Threshold Selection

I evaluated seven thresholds between 0.20 and 0.80 using the validation set. The threshold of 0.60 gave the highest validation F1 score of approximately 0.831169, so I selected it before evaluating the test set.

On the test set, the selected threshold produced precision of 0.903226, recall of 0.666667, and F1 of 0.767123. Compared with the default threshold, precision increased, but recall and F1 decreased. False positives decreased from 4 to 3, while false negatives increased from 11 to 14.

I found this comparison useful because it showed the trade-off involved in threshold selection. Although 0.60 performed best on validation F1, it did not perform better than 0.50 on test F1. I kept the selected threshold rather than changing it based on test results, because the test set should remain independent of the model-selection process. The log-loss value stayed the same under both thresholds because changing the threshold changes the class predictions, not the predicted probabilities.

## 6. Limitations and Conclusion

Although the results were encouraging, I would not consider this model ready for clinical use. A false negative is particularly important in this situation because a malignant case classified as benign could delay further medical assessment. There are also limitations to my analysis. The model uses only two features, and it has not been tested on an independent external dataset. Its calibration has also not been verified for clinical decision-making.

Overall, this project helped me understand the complete classification process, from calculating probabilities to selecting a decision threshold and evaluating the final predictions. It also showed me why a model should be evaluated using several performance measures rather than accuracy alone.

---

## Research Process 

One important decision I made during this project was to code malignant cases as the positive class. This made it easier to interpret the model's predictions and understand the consequences of false negatives. I also used stratified splitting and a fixed random seed so that the results could be reproduced.

When implementing logistic regression with NumPy, I used gradient descent to update the coefficients and monitored the training loss. I then compared my results with scikit-learn's implementation. The very small differences in the coefficients and predicted probabilities helped me check that my implementation was working as expected.

Another important decision was to use the validation set for threshold selection instead of using the test set. I selected 0.60 because it produced the highest validation F1 score. Even though the default threshold performed better on test F1, I kept the selected threshold unchanged. This helped me maintain a fair separation between model selection and final evaluation.

I also organized the required output files in a `reports` folder and used a project-relative output path so that the script could save its results in the correct location when run from the project folder.