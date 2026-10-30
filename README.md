# Breast Mass Classification – Logistic Regression

## 1. Project Overview

This project demonstrates the implementation and evaluation of logistic regression for breast mass classification using the Wisconsin Diagnostic Breast Cancer dataset.

The analysis uses two features, mean radius and mean texture, to classify observations as malignant or benign. It includes a NumPy implementation using gradient descent, a comparison with scikit-learn, baseline evaluation, and classification threshold selection.

## 2. Project Structure

```text
Breast-Mass-Classification-Logistic-Regression/
│
├── logistic_regression_analysis.py   # Main analysis script
├── requirements.txt                  # Required Python packages
├── README.md                          # Project reproduction guide
├── memo.md                            # Written project report
└── reports/
    ├── threshold_results.csv          # Validation threshold results
    ├── assignment_results.csv         # Model evaluation results
    └── training_loss.png              # Training loss visualization
```

## 3. Prerequisites

Before starting, make sure you have installed:

- Python 3.10 or later
- Git
- Visual Studio Code (optional)

Installation links:

- [Python](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code](https://code.visualstudio.com/download)

## 4. Clone the GitHub Repository

Open a terminal or command prompt and run:

```bash
git clone https://github.com/Vaishnaviram1026/Breast-Mass-Classification-Logistic-Regression.git
```

Navigate to the project directory:

```bash
cd Breast-Mass-Classification-Logistic-Regression
```

## 5. Create and Activate a Virtual Environment

A virtual environment keeps the project's Python packages separate from other Python projects.

### Windows – PowerShell

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

**If PowerShell blocks activation:**

Windows may prevent PowerShell scripts from running because of its execution policy. If you receive an error stating that script execution is disabled, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If prompted, confirm the change. Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

The `-Scope Process` option applies only to the current PowerShell session and does not permanently change the computer's execution policy.

### Windows – Command Prompt (Alternative)

If you prefer using Command Prompt instead of PowerShell, create the environment:

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate.bat
```

### macOS – Terminal

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

If Python is not found, check whether Python 3 is installed:

```bash
python3 --version
```

If necessary, install it from [python.org](https://www.python.org/downloads/).

### Confirm Activation

After successful activation, the environment name should appear at the beginning of your terminal prompt:

```text
(.venv)
```

## 6. Install the Required Packages

Make sure the virtual environment is activated before installing the dependencies.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The project uses the following Python libraries:

- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## 7. Run the Python Script

From the main project directory, execute:

```bash
python logistic_regression_analysis.py
```

Wait for the script to finish running.

The script performs the analysis and generates the required output files automatically.

**Dataset download:** No separate dataset download is required. The Wisconsin Diagnostic Breast Cancer dataset is loaded directly from scikit-learn.

## 8. Locate the Generated Output Files

After successful execution, open the `reports` folder.

The following files should be available:

| Output file | Description |
|---|---|
| `threshold_results.csv` | Validation results for the seven classification thresholds |
| `assignment_results.csv` | Model evaluation results for validation and test datasets |
| `training_loss.png` | Plot showing training loss across epochs |

The script automatically creates the `reports` folder if it does not already exist.

## 9. Troubleshooting

### Python is not recognized

Check that Python is installed correctly and added to your system PATH. Restart the terminal after installation.

### Virtual environment activation is blocked on Windows

In PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, use Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### Required package or module is missing

Make sure the virtual environment is activated, then run:

```bash
python -m pip install -r requirements.txt
```

### Output files are missing

Check that the Python script completed without errors. Confirm that you are running it from the cloned project directory and check the `reports` folder.

---

**Tested environment:** Windows, Python 3.14.3, Visual Studio Code.

The macOS instructions are provided for reproduction; they have not been independently tested as part of this project.
