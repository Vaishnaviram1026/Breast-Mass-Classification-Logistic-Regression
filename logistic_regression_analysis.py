
# ============================================================
# PROJECT: Breast Mass Classification
# Course: MSBA265
# Part 1: Connect the Equation to a Prediction
# ============================================================

import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Logistic regression calculation function
def logistic_prediction(beta_0, beta_1, x, threshold):

    # Step 1: Calculate the score
    z = beta_0 + beta_1 * x

    # Step 2: Convert score into probability using sigmoid
    p = 1 / (1 + np.exp(-z))

    # Step 3: Predict the class using the threshold
    predicted_class = 1 if p >= threshold else 0

    return z, p, predicted_class


# Four required exploration cases
cases = [
    {"case": "A", "beta_0": -1.0, "beta_1":  0.8, "x": 2.0, "threshold": 0.50},
    {"case": "B", "beta_0": -1.0, "beta_1":  0.8, "x": 2.0, "threshold": 0.70},
    {"case": "C", "beta_0": -1.0, "beta_1": -0.8, "x": 2.0, "threshold": 0.50},
    {"case": "D", "beta_0":  1.0, "beta_1":  0.0, "x": 2.0, "threshold": 0.50},
]


# Display results
print("\nPART 1: LOGISTIC REGRESSION EXPLORATION")
print("=" * 75)

print(f"{'Case':<10}{'Score (z)':<15}{'Probability':<18}{'Threshold':<15}{'Prediction'}")
print("-" * 75)

for case in cases:

    z, p, prediction = logistic_prediction(
        case["beta_0"],
        case["beta_1"],
        case["x"],
        case["threshold"]
    )

    print(
        f"{case['case']:<10}"
        f"{z:<15.4f}"
        f"{p:<18.4f}"
        f"{case['threshold']:<15.2f}"
        f"{prediction}"
    )


# Worked calculation for Case A
print("\nWORKED CALCULATION: CASE A")
print("=" * 75)

z_a = -1.0 + (0.8 * 2.0)
p_a = 1 / (1 + np.exp(-z_a))

print("Score:")
print("z = -1.0 + (0.8 * 2.0)")
print(f"z = {z_a:.4f}")

print("\nSigmoid probability:")
print("p = 1 / (1 + exp(-z))")
print(f"p = {p_a:.4f}")

print("\nClassification:")
print("Since probability >= 0.50, predicted class = 1")


# Explanations
print("\nINTERPRETATION")
print("=" * 75)

print("Case A vs Case B:")
print("The coefficients and probability remain the same.")
print("Only the threshold increases from 0.50 to 0.70.")
print("Therefore, the predicted class changes from 1 to 0.")

print("\nCase C:")
print("Reversing the coefficient from +0.8 to -0.8")
print("reduces the score and probability.")

print("\nCase D:")
print("The coefficient is zero, so the input has no effect.")
print("The probability remains constant for every value of x.")


# ============================================================
# PART 2: PREPARE THE DATASET
# ============================================================

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print("\n\nPART 2: DATASET PREPARATION")
print("=" * 75)


# 1. Load the Wisconsin Diagnostic Breast Cancer dataset
data = load_breast_cancer()

print("\nDataset loaded successfully!")

print(data.DESCR)


# 2. Select exactly two required features, in the required order
feature_names = ["mean radius", "mean texture"]

X = pd.DataFrame(
    data.data[:, [
        list(data.feature_names).index("mean radius"),
        list(data.feature_names).index("mean texture")
    ]],
    columns=feature_names
)


# 3. Recode target: malignant = 1, benign = 0
y = (data.target == 0).astype(int)


# 4. Display dataset information
print("\nSelected features:")
print(feature_names)

print("\nNumber of observations:", len(X))

print("\nMissing values:")
print(X.isnull().sum())

print("\nOverall class counts:")
print("0 = Benign")
print("1 = Malignant")
print(pd.Series(y).value_counts().sort_index())


# 5. Acknowledge dataset source
print("\nDataset source:")
print("Wisconsin Diagnostic Breast Cancer dataset")
print("Loaded using sklearn.datasets.load_breast_cancer()")
print("Original dataset source: UCI Machine Learning Repository")


# 6. Split row indices into development and test
all_indices = np.arange(len(y))

development_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# 7. Split development into training and validation
train_indices, validation_indices = train_test_split(
    development_indices,
    test_size=0.25,
    stratify=y[development_indices],
    random_state=42
)


# 8. Create the three partitions
X_train = X.iloc[train_indices].to_numpy()
X_validation = X.iloc[validation_indices].to_numpy()
X_test = X.iloc[test_indices].to_numpy()

y_train = y[train_indices]
y_validation = y[validation_indices]
y_test = y[test_indices]


# 9. Print partition sizes and class counts
def show_partition(name, indices, targets):

    print(f"\n{name} partition")
    print("-" * 40)

    print("Rows:", len(indices))
    print("Benign (0):", np.sum(targets == 0))
    print("Malignant (1):", np.sum(targets == 1))


show_partition("Training", train_indices, y_train)
show_partition("Validation", validation_indices, y_validation)
show_partition("Test", test_indices, y_test)


# 10. Verify no overlap between partitions
assert len(set(train_indices) & set(validation_indices)) == 0
assert len(set(train_indices) & set(test_indices)) == 0
assert len(set(validation_indices) & set(test_indices)) == 0

# Verify that all rows are included
combined_indices = np.concatenate([
    train_indices,
    validation_indices,
    test_indices
])

assert len(combined_indices) == len(y)
assert len(np.unique(combined_indices)) == len(y)

print("\nPartition overlap check: PASSED")
print("All rows accounted for: PASSED")


# 11. Standardize using training data ONLY
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)

# ============================================================
# PART 3: LOGISTIC REGRESSION FROM SCRATCH USING NUMPY
# ============================================================

print("\n\nPART 3: NUMPY LOGISTIC REGRESSION")
print("=" * 75)


# 1. Add an intercept column of ones
X_train_augmented = np.column_stack([
    np.ones(len(X_train_scaled)),
    X_train_scaled
])


# 2. Define a numerically stable sigmoid function
def sigmoid(z):
    z = np.asarray(z)
    result = np.empty_like(z, dtype=float)

    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1 + exp_z)

    return result


# 3. Define binary cross-entropy loss
def binary_cross_entropy(y_true, probabilities):
    epsilon = 1e-15

    probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )

    return loss


# 4. Initialize coefficients to zero
beta = np.zeros(X_train_augmented.shape[1])

learning_rate = 0.1
epochs = 20000

loss_history = []


# 5. Record initial loss before training
initial_probabilities = sigmoid(X_train_augmented @ beta)

initial_loss = binary_cross_entropy(
    y_train,
    initial_probabilities
)

loss_history.append(initial_loss)


# 6. Full-batch gradient descent
for epoch in range(epochs):

    probabilities = sigmoid(X_train_augmented @ beta)

    gradient = (
        X_train_augmented.T @ (probabilities - y_train)
    ) / len(y_train)

    beta = beta - learning_rate * gradient

    updated_probabilities = sigmoid(X_train_augmented @ beta)

    updated_loss = binary_cross_entropy(
        y_train,
        updated_probabilities
    )

    loss_history.append(updated_loss)


# 7. Display training results
print("\nTraining completed!")

print("Learning rate:", learning_rate)
print("Number of epochs:", epochs)

print("\nInitial loss:", initial_loss)
print("Final loss:", loss_history[-1])

print("\nLearned coefficients (standardized feature units):")
print("Intercept:", beta[0])
print("Mean radius coefficient:", beta[1])
print("Mean texture coefficient:", beta[2])


# 8. Plot training loss
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("NumPy Logistic Regression Training Loss")

plt.grid(True)
plt.tight_layout()

plt.savefig(REPORTS_DIR / "training_loss.png", dpi=300)

plt.show()

# ============================================================
# PART 4: SCIKIT-LEARN LOGISTIC REGRESSION COMPARISON
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

print("\n\nPART 4: SCIKIT-LEARN COMPARISON")
print("=" * 75)


# 1. Fit scikit-learn logistic regression
sklearn_model = LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)

sklearn_model.fit(X_train_scaled, y_train)


# 2. Get scikit-learn validation probabilities
sklearn_validation_all = sklearn_model.predict_proba(
    X_validation_scaled
)

# Identify the probability column for malignant class (1)
positive_class_column = list(sklearn_model.classes_).index(1)

sklearn_validation_probabilities = (
    sklearn_validation_all[:, positive_class_column]
)


# 3. Get NumPy model validation probabilities
X_validation_augmented = np.column_stack([
    np.ones(len(X_validation_scaled)),
    X_validation_scaled
])

numpy_validation_probabilities = sigmoid(
    X_validation_augmented @ beta
)


# 4. Compare coefficients
print("\nNumPy model coefficients (standardized units):")
print("Intercept:", beta[0])
print("Mean radius:", beta[1])
print("Mean texture:", beta[2])

print("\nScikit-learn model coefficients (standardized units):")
print("Intercept:", sklearn_model.intercept_[0])
print("Mean radius:", sklearn_model.coef_[0][0])
print("Mean texture:", sklearn_model.coef_[0][1])


# 5. Compare validation probabilities
mean_absolute_difference = np.mean(
    np.abs(
        numpy_validation_probabilities
        - sklearn_validation_probabilities
    )
)

print("\nMean absolute difference in validation probabilities:")
print(mean_absolute_difference)


# 6. Calculate validation log loss
numpy_validation_log_loss = log_loss(
    y_validation,
    numpy_validation_probabilities
)

sklearn_validation_log_loss = log_loss(
    y_validation,
    sklearn_validation_probabilities
)

print("\nNumPy validation log loss:", numpy_validation_log_loss)
print("Scikit-learn validation log loss:", sklearn_validation_log_loss)


# 7. Fit prior-probability baseline
baseline_model = DummyClassifier(strategy="prior")

baseline_model.fit(X_train_scaled, y_train)

baseline_validation_all = baseline_model.predict_proba(
    X_validation_scaled
)

baseline_validation_probabilities = (
    baseline_validation_all[:, positive_class_column]
)

baseline_validation_log_loss = log_loss(
    y_validation,
    baseline_validation_probabilities
)

print("\nBaseline validation log loss:", baseline_validation_log_loss)


# ============================================================
# PART 5: THRESHOLD SELECTION AND FINAL MODEL EVALUATION
# ============================================================

import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

print("\n\nPART 5: THRESHOLD SELECTION AND MODEL EVALUATION")
print("=" * 75)


# ------------------------------------------------------------
# 1. Define the seven required classification thresholds
# ------------------------------------------------------------

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


# ------------------------------------------------------------
# 2. Calculate validation metrics for each threshold
#    IMPORTANT: Threshold selection uses validation data only.
# ------------------------------------------------------------

threshold_results = []

for threshold in thresholds:

    predictions = (
        numpy_validation_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_validation, predictions)

    precision = precision_score(
        y_validation, predictions, zero_division=0
    )

    recall = recall_score(
        y_validation, predictions, zero_division=0
    )

    f1 = f1_score(
        y_validation, predictions, zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ------------------------------------------------------------
# 3. Display threshold results
# ------------------------------------------------------------

threshold_df = pd.DataFrame(threshold_results)

print("\nVALIDATION THRESHOLD RESULTS")
print(threshold_df.to_string(index=False))


# ------------------------------------------------------------
# 4. Select the threshold with the highest F1.
#    If tied, select the smallest threshold.
# ------------------------------------------------------------

best_result = min(
    threshold_results,
    key=lambda row: (-row["f1"], row["threshold"])
)

selected_threshold = best_result["threshold"]

print("\nSelected threshold:", selected_threshold)
print("Selected validation F1:", best_result["f1"])


# ------------------------------------------------------------
# 5. Freeze the threshold before test evaluation
# ------------------------------------------------------------

print("\nThreshold selection completed using validation data only.")
print("The selected threshold is now frozen.")


# ------------------------------------------------------------
# 6. Define a function to get class 1 probabilities
# ------------------------------------------------------------

def get_positive_probability(model, features):

    probabilities = model.predict_proba(features)

    positive_column = list(model.classes_).index(1)

    return probabilities[:, positive_column]


# ------------------------------------------------------------
# 7. Prepare probabilities for all models
# ------------------------------------------------------------

# Validation probabilities
numpy_val_probs = numpy_validation_probabilities

sklearn_val_probs = sklearn_validation_probabilities

baseline_val_probs = baseline_validation_probabilities


# Test probabilities are generated only AFTER threshold selection.
X_test_augmented = np.column_stack([
    np.ones(len(X_test_scaled)),
    X_test_scaled
])

numpy_test_probs = sigmoid(
    X_test_augmented @ beta
)

sklearn_test_probs = get_positive_probability(
    sklearn_model,
    X_test_scaled
)

baseline_test_probs = get_positive_probability(
    baseline_model,
    X_test_scaled
)


# ------------------------------------------------------------
# 8. Define evaluation function
# ------------------------------------------------------------

def evaluate_model(
    y_true,
    probabilities,
    threshold,
    split_name,
    model_name,
    policy_name
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "split": split_name,
        "model": model_name,
        "policy": policy_name,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        )
    }


# ------------------------------------------------------------
# 9. Evaluate five configurations on validation and test
# ------------------------------------------------------------

configurations = [
    ("baseline", "default", 0.50,
     baseline_val_probs, baseline_test_probs),

    ("numpy", "default", 0.50,
     numpy_val_probs, numpy_test_probs),

    ("numpy", "selected", selected_threshold,
     numpy_val_probs, numpy_test_probs),

    ("sklearn", "default", 0.50,
     sklearn_val_probs, sklearn_test_probs),

    ("sklearn", "selected", selected_threshold,
     sklearn_val_probs, sklearn_test_probs)
]


all_results = []

for model_name, policy_name, threshold, val_probs, test_probs in configurations:

    # Validation evaluation
    all_results.append(
        evaluate_model(
            y_validation,
            val_probs,
            threshold,
            "validation",
            model_name,
            policy_name
        )
    )

    # Test evaluation
    all_results.append(
        evaluate_model(
            y_test,
            test_probs,
            threshold,
            "test",
            model_name,
            policy_name
        )
    )


# ------------------------------------------------------------
# 10. Display final results
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

print("\nFINAL MODEL EVALUATION RESULTS")
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 11. Save required CSV files
# ------------------------------------------------------------

threshold_df.to_csv(
    REPORTS_DIR / "threshold_results.csv",
    index=False,
    float_format="%.8f"
)

results_df.to_csv(
    REPORTS_DIR / "assignment_results.csv",
    index=False,
    float_format="%.8f"
)

print("\nSaved:")
print("reports/threshold_results.csv")
print("reports/assignment_results.csv")



# ============================================================
# PART 1: OPTIONAL MATPLOTLIB INTERACTIVE EXPLORER
# ============================================================

def launch_logistic_gui():

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # Initial values: Case A
    x_curve = np.linspace(-10, 10, 500)

    initial_beta_0 = -1.0
    initial_beta_1 = 0.8
    initial_x = 2.0
    initial_threshold = 0.50

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(figsize=(13, 10))

    fig.suptitle(
        "Interactive Logistic Regression Explorer",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # Graph area: extra bottom margin for x-axis label
    ax = fig.add_axes([0.13, 0.55, 0.76, 0.34])

    ax.set_title("Sigmoid Probability Curve", fontsize=12, pad=12)
    ax.set_xlabel("Input value (x)", labelpad=12)
    ax.set_ylabel("Predicted probability", labelpad=12)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Logistic regression calculation
    # --------------------------------------------------------

    def calculate_probability(beta_0, beta_1, x):

        z = beta_0 + beta_1 * x
        probability = 1 / (1 + np.exp(-z))

        return z, probability

    # Initial curve
    initial_scores = (
        initial_beta_0 + initial_beta_1 * x_curve
    )

    initial_probabilities = (
        1 / (1 + np.exp(-initial_scores))
    )

    curve, = ax.plot(
        x_curve,
        initial_probabilities,
        linewidth=2.5,
        label="Sigmoid curve"
    )

    threshold_line = ax.axhline(
        initial_threshold,
        linestyle="--",
        linewidth=1.5,
        label="Classification threshold"
    )

    input_line = ax.axvline(
        initial_x,
        linestyle=":",
        linewidth=1.5,
        label="Current input"
    )

    _, initial_probability = calculate_probability(
        initial_beta_0,
        initial_beta_1,
        initial_x
    )

    probability_point, = ax.plot(
        [initial_x],
        [initial_probability],
        marker="o",
        markersize=9,
        linestyle="None",
        label="Current probability"
    )

    ax.legend(loc="lower right", fontsize=9)

    # --------------------------------------------------------
    # Calculation results: dedicated area
    # --------------------------------------------------------

    result_text = fig.text(
        0.50,
        0.475,
        "",
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.8,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "white",
            "edgecolor": "gray"
        }
    )

    # --------------------------------------------------------
    # Slider layout: labels on the left
    # --------------------------------------------------------

    slider_rows = [
        ("Intercept (beta 0)", -5.0, 5.0, initial_beta_0),
        ("Coefficient (beta 1)", -2.0, 2.0, initial_beta_1),
        ("Input value (x)", -10.0, 10.0, initial_x),
        ("Threshold", 0.05, 0.95, initial_threshold)
    ]

    slider_objects = []

    slider_y_positions = [0.365, 0.295, 0.225, 0.155]

    for (label, minimum, maximum, initial), y_position in zip(
        slider_rows,
        slider_y_positions
    ):

        # Separate label area on the left
        fig.text(
            0.08,
            y_position + 0.005,
            label,
            fontsize=10,
            ha="left",
            va="center"
        )

        # Slider begins well to the right of its label
        slider_ax = fig.add_axes([
            0.36,
            y_position - 0.005,
            0.48,
            0.025
        ])

        slider = Slider(
            slider_ax,
            "",
            minimum,
            maximum,
            valinit=initial
        )

        # Numeric value displayed at the right
        slider.valtext.set_position((1.08, 0.5))

        slider_objects.append(slider)

    beta_0_slider = slider_objects[0]
    beta_1_slider = slider_objects[1]
    x_slider = slider_objects[2]
    threshold_slider = slider_objects[3]

    # --------------------------------------------------------
    # Update graph and calculations
    # --------------------------------------------------------

    def update_graph(value=None):

        beta_0 = beta_0_slider.val
        beta_1 = beta_1_slider.val
        x = x_slider.val
        threshold = threshold_slider.val

        z, probability = calculate_probability(
            beta_0,
            beta_1,
            x
        )

        predicted_class = (
            1 if probability >= threshold else 0
        )

        scores = beta_0 + beta_1 * x_curve

        probabilities = (
            1 / (1 + np.exp(-scores))
        )

        curve.set_ydata(probabilities)

        threshold_line.set_ydata(
            [threshold, threshold]
        )

        input_line.set_xdata([x, x])

        probability_point.set_data(
            [x],
            [probability]
        )

        result_text.set_text(
            f"Score (z): {z:.4f}       "
            f"Probability: {probability:.4f}\n"
            f"Threshold: {threshold:.2f}       "
            f"Predicted class: {predicted_class}"
        )

        fig.canvas.draw_idle()

    beta_0_slider.on_changed(update_graph)
    beta_1_slider.on_changed(update_graph)
    x_slider.on_changed(update_graph)
    threshold_slider.on_changed(update_graph)

    # --------------------------------------------------------
    # Four assignment case buttons
    # --------------------------------------------------------

    case_settings = [
        (-1.0, 0.8, 2.0, 0.50),
        (-1.0, 0.8, 2.0, 0.70),
        (-1.0, -0.8, 2.0, 0.50),
        (1.0, 0.0, 2.0, 0.50)
    ]

    button_width = 0.15
    button_gap = 0.04

    total_width = (
        4 * button_width + 3 * button_gap
    )

    start_position = (1 - total_width) / 2

    case_buttons = []

    def select_case(settings):

        beta_0, beta_1, x, threshold = settings

        beta_0_slider.set_val(beta_0)
        beta_1_slider.set_val(beta_1)
        x_slider.set_val(x)
        threshold_slider.set_val(threshold)

    for index, settings in enumerate(case_settings):

        left = (
            start_position
            + index * (button_width + button_gap)
        )

        button_ax = fig.add_axes([
            left,
            0.055,
            button_width,
            0.045
        ])

        button = Button(
            button_ax,
            f"Case {chr(65 + index)}"
        )

        button.on_clicked(
            lambda event, values=settings: select_case(values)
        )

        case_buttons.append(button)

    # Footer
    fig.text(
        0.50,
        0.015,
        "Coefficients are manually adjusted; this explorer does not train a model.",
        ha="center",
        fontsize=9
    )

    update_graph()

    plt.show()


# ============================================================
# OPTIONAL GUI CONTROL
# ============================================================

RUN_GUI = False  # Set to True to launch the interactive GUI 

if RUN_GUI:
    launch_logistic_gui()
