# Run a Pipeline Job — Detailed Azure ML Lab Guide

## 1. Lab Title

**Run scripts as a pipeline job using Azure Machine Learning**

---

## 2. Objective

This notebook demonstrates how to build an Azure Machine Learning pipeline from multiple command components.

The pipeline has two main steps:

1. **Prepare the data**
   - Read the diabetes data.
   - Remove missing values.
   - Normalize selected numeric columns.
   - Save the prepared data.

2. **Train the model**
   - Read the prepared data.
   - Split it into training and test sets.
   - Train a Logistic Regression model.
   - Evaluate the model using Accuracy, AUC, and an ROC curve.
   - Save the trained model.

The notebook defines each step as a YAML component, loads those components, connects them into a pipeline, configures pipeline-level settings, and submits the pipeline as an Azure ML job.

---

# 3. Overall Pipeline Flow

```text
Registered diabetes data asset
          |
          v
   prep_data component
          |
          | cleaned + normalized data
          v
train_logistic_regression component
          |
          +--------------------+
          |                    |
          v                    v
 transformed data          trained model
```

The important concept is that the **output of one component becomes the input of the next component**.

---

# 4. Notebook Structure

The notebook follows this sequence:

1. Verify `azure-ai-ml`
2. Authenticate with Azure
3. Connect to the Azure ML workspace
4. Create the `src` folder
5. Create `prep-data.py`
6. Create `train-model.py`
7. Define `prep-data.yml`
8. Define `train-model.yml`
9. Load the components
10. Build the pipeline
11. Inspect the pipeline configuration
12. Modify output mode, compute, and datastore
13. Submit the pipeline job

---

# 5. Before You Start

The notebook states that the latest version of the **`azure-ai-ml`** package is required.

If it is not installed, the notebook instructs you to install it with:

```bash
pip install azure-ai-ml
```

The notebook itself first checks whether the package is installed.

---

# 6. Cell-by-Cell Explanation

## Cell 1 — Notebook Introduction

The notebook starts with:

```text
# Run scripts as a pipeline job
```

The notebook explains that a pipeline allows multiple steps to be grouped into one workflow.

Each pipeline step is represented by a **component**.

A component corresponds to a Python script that Azure ML knows how to execute.

The component is described using a YAML file containing information such as:

- the component name
- inputs
- outputs
- Python source code
- environment
- command used to execute the script

### Key idea

```text
Python script
     +
Component YAML
     =
Azure ML component
```

Multiple components can then be connected to form a pipeline.

---

# 7. Cell 2 — Check Azure ML SDK

The notebook contains:

```python
pip show azure-ai-ml
```

### `pip show`

`pip show` displays information about an installed Python package.

### `azure-ai-ml`

This is the Azure Machine Learning Python SDK package.

The command can show information such as:

- package name
- installed version
- package location
- dependencies

### Why this cell exists

The notebook wants to verify that the required Azure ML SDK is available before executing the rest of the notebook.

---

# 8. Connect to the Azure ML Workspace

The notebook next explains that it needs to connect to the Azure ML workspace.

It uses:

- subscription information
- resource group information
- workspace information

Because the notebook is running in an Azure ML-managed compute environment, it uses the workspace configuration available to the environment.

---

# 9. Cell 3 — Import Azure Authentication Classes

```python
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
```

## `DefaultAzureCredential`

```python
from azure.identity import DefaultAzureCredential
```

`DefaultAzureCredential` provides a convenient way to authenticate with Azure.

It can attempt several credential sources depending on the environment.

This is useful because the same notebook can work in different Azure development environments without hard-coding credentials.

---

## `InteractiveBrowserCredential`

```python
from azure.identity import InteractiveBrowserCredential
```

This credential allows authentication through a browser.

The notebook uses it as a fallback if `DefaultAzureCredential` cannot successfully obtain a token.

---

## `MLClient`

```python
from azure.ai.ml import MLClient
```

`MLClient` is the main Azure ML SDK client used in this notebook.

It provides access to Azure ML resources and operations, including jobs.

---

# 10. Authentication Logic

The notebook contains:

```python
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()
```

Let's examine every part.

---

## `try`

```python
try:
```

Python's `try` block contains code that might raise an exception.

The notebook attempts to authenticate using the default Azure credential mechanism.

---

## Create the default credential

```python
credential = DefaultAzureCredential()
```

An authentication object is created and assigned to:

```python
credential
```

---

## Request an Azure management token

```python
credential.get_token("https://management.azure.com/.default")
```

This attempts to obtain an Azure access token.

The requested scope is:

```text
https://management.azure.com/.default
```

This is the Azure Resource Manager management scope.

### Why call `get_token()` explicitly?

The notebook uses this call as a quick authentication test.

If the credential cannot obtain a token, an exception can occur.

---

## `except`

```python
except Exception as ex:
```

If authentication through `DefaultAzureCredential` fails, Python enters this block.

`Exception` is the general Python exception class.

`ex` stores the exception object.

The notebook does not print the exception.

---

## Browser fallback

```python
credential = InteractiveBrowserCredential()
```

If the default credential mechanism fails, the notebook creates an interactive browser credential.

This allows the user to authenticate through a browser.

---

# 11. Cell 4 — Create the ML Client

```python
# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)
```

## Comment

```python
# Get a handle to workspace
```

The comment explains that the code creates a client connected to the workspace.

---

## `MLClient.from_config()`

```python
MLClient.from_config(credential=credential)
```

This creates an `MLClient` using the workspace configuration.

The notebook passes:

```python
credential=credential
```

so Azure ML uses the credential created in the previous cell.

The resulting client is stored in:

```python
ml_client
```

### Why `ml_client` matters

Later, the notebook uses:

```python
ml_client.jobs.create_or_update(...)
```

to submit the pipeline job.

---

# 12. Create the Scripts

The notebook creates two pipeline steps:

### Step 1 — Prepare data

The script:

```text
prep-data.py
```

performs:

- reading data
- removing missing values
- normalization
- saving the transformed data

### Step 2 — Train model

The script:

```text
train-model.py
```

performs:

- reading prepared data
- splitting data
- training Logistic Regression
- calculating Accuracy
- calculating AUC
- creating an ROC curve
- saving the model

---

# 13. Cell 5 — Create the `src` Folder

```python
import os

# create a folder for the script files
script_folder = 'src'
os.makedirs(script_folder, exist_ok=True)
print(script_folder, 'folder created')
```

---

## Import `os`

```python
import os
```

The `os` module provides operating-system-related functionality.

Here it is used to create a directory.

---

## Define the folder name

```python
script_folder = 'src'
```

A variable named:

```python
script_folder
```

is assigned the string:

```text
src
```

The notebook will store both Python scripts inside this directory.

---

## Create the directory

```python
os.makedirs(script_folder, exist_ok=True)
```

### `os.makedirs()`

Creates a directory.

### Parameter: `script_folder`

Specifies the directory to create:

```text
src
```

### Parameter: `exist_ok=True`

This is important.

It means:

> Do not raise an error if the directory already exists.

Without `exist_ok=True`, attempting to create an existing directory could produce an error.

---

## Print confirmation

```python
print(script_folder, 'folder created')
```

This prints a confirmation message such as:

```text
src folder created
```

---

# 14. Cell 6 — Create `prep-data.py`

The notebook uses:

```python
%%writefile $script_folder/prep-data.py
```

This is a Jupyter/IPython magic command.

It writes the following cell contents into:

```text
src/prep-data.py
```

The `$script_folder` part refers to the variable:

```python
script_folder = 'src'
```

Therefore:

```text
$script_folder/prep-data.py
```

becomes:

```text
src/prep-data.py
```

---

# 15. `prep-data.py` — Imports

The generated script starts with:

```python
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
```

---

## `argparse`

```python
import argparse
```

`argparse` allows the Python script to receive command-line arguments.

For example:

```text
--input_data
--output_data
```

These are supplied by the Azure ML component command.

---

## pandas

```python
import pandas as pd
```

Pandas is used for tabular data processing.

The conventional alias is:

```python
pd
```

---

## NumPy

```python
import numpy as np
```

NumPy provides numerical computing functionality.

In this particular `prep-data.py` script, the notebook imports NumPy but the shown functions do not directly use `np`.

---

## `Path`

```python
from pathlib import Path
```

`Path` provides object-oriented handling of filesystem paths.

It is later used to construct the output file path.

---

## `MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler
```

`MinMaxScaler` performs feature scaling.

It transforms values into a specified range, with the default range being:

```text
0 to 1
```

---

# 16. `prep-data.py` — `main()`

```python
def main(args):
```

This defines the main workflow of the preparation script.

`args` contains command-line arguments parsed by `argparse`.

---

## Read data

```python
df = get_data(args.input_data)
```

The input data path comes from:

```python
args.input_data
```

That path is passed to:

```python
get_data()
```

The returned DataFrame is stored as:

```python
df
```

---

## Clean data

```python
cleaned_data = clean_data(df)
```

The raw DataFrame is passed to:

```python
clean_data()
```

The returned DataFrame is stored as:

```python
cleaned_data
```

---

## Normalize data

```python
normalized_data = normalize_data(cleaned_data)
```

The cleaned data is passed to:

```python
normalize_data()
```

The normalized DataFrame is stored in:

```python
normalized_data
```

---

## Save the output

```python
output_df = normalized_data.to_csv((Path(args.output_data) / "diabetes.csv"), index = False)
```

This writes the transformed data to a CSV file.

### `Path(args.output_data)`

Creates a `Path` object for the output directory.

### `/ "diabetes.csv"`

The `/` operator is used by `Path` to construct a child path.

Conceptually:

```text
output directory
       +
diabetes.csv
```

### `to_csv()`

Writes the DataFrame to CSV.

### `index=False`

Prevents pandas from writing the DataFrame index as an additional CSV column.

### `output_df`

The result of `to_csv()` is assigned to `output_df`.

The notebook does not subsequently use this variable.

---

# 17. `get_data()`

```python
def get_data(path):
    df = pd.read_csv(path)

    # Count the rows and print the result
    row_count = (len(df))
    print('Preparing {} rows of data'.format(row_count))

    return df
```

---

## Function definition

```python
def get_data(path):
```

The function accepts:

```text
path
```

which represents the input data file.

---

## Read CSV

```python
df = pd.read_csv(path)
```

`pd.read_csv()` reads the CSV file into a pandas DataFrame.

The resulting DataFrame is assigned to:

```python
df
```

---

## Count rows

```python
row_count = (len(df))
```

`len(df)` returns the number of rows.

The value is stored in:

```python
row_count
```

---

## Print row count

```python
print('Preparing {} rows of data'.format(row_count))
```

The `.format()` method inserts the row count into the message.

For example:

```text
Preparing 1000 rows of data
```

---

## Return DataFrame

```python
return df
```

The function returns the DataFrame to the caller.

---

# 18. `clean_data()`

```python
def clean_data(df):
    df = df.dropna()

    return df
```

---

## Function

```python
def clean_data(df):
```

Receives the DataFrame.

---

## Remove missing values

```python
df = df.dropna()
```

`dropna()` removes rows containing missing values.

The cleaned DataFrame replaces the previous value of `df`.

---

## Return

```python
return df
```

Returns the cleaned DataFrame.

---

# 19. `normalize_data()`

```python
def normalize_data(df):
    scaler = MinMaxScaler()
    num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
```

---

## Create scaler

```python
scaler = MinMaxScaler()
```

Creates a Min-Max scaling object.

By default, MinMaxScaler maps each selected feature to the range:

```text
0 to 1
```

---

## Define numeric columns

```python
num_cols = [
    'Pregnancies',
    'PlasmaGlucose',
    'DiastolicBloodPressure',
    'TricepsThickness',
    'SerumInsulin',
    'BMI',
    'DiabetesPedigree'
]
```

These are the columns that will be normalized.

Notice that:

```text
Age
```

is not included in this particular normalization list.

The target:

```text
Diabetic
```

is also not included.

---

## Fit and transform

```python
df[num_cols] = scaler.fit_transform(df[num_cols])
```

This performs two operations.

### `fit`

The scaler learns the minimum and maximum values of each selected column.

### `transform`

The learned values are used to scale the data.

### `fit_transform`

Combines both operations.

The normalized values replace the original values in:

```python
df[num_cols]
```

---

## Return

```python
return df
```

Returns the normalized DataFrame.

---

# 20. `parse_args()`

```python
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", dest='input_data',
                        type=str)
    parser.add_argument("--output_data", dest='output_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args
```

This function allows the Azure ML component command to pass values into the Python script.

---

## Create parser

```python
parser = argparse.ArgumentParser()
```

Creates an argument parser.

---

## Input argument

```python
parser.add_argument("--input_data", dest='input_data',
                    type=str)
```

Defines the command-line argument:

```text
--input_data
```

### `dest='input_data'`

The value will be available as:

```python
args.input_data
```

### `type=str`

The argument is treated as a string.

---

## Output argument

```python
parser.add_argument("--output_data", dest='output_data',
                    type=str)
```

Defines:

```text
--output_data
```

Its value is available through:

```python
args.output_data
```

---

## Parse

```python
args = parser.parse_args()
```

Reads the arguments supplied when the script is executed.

---

## Return

```python
return args
```

Returns the parsed arguments.

---

# 21. Script Entry Point

```python
if __name__ == "__main__":
```

This is a standard Python pattern.

It means the following code runs when the file is executed directly.

---

## Print spacing

```python
print("\n\n")
```

Prints blank lines.

---

## Print separator

```python
print("*" * 60)
```

Creates a string containing 60 asterisks.

It is used as a visual separator in logs.

---

## Parse arguments

```python
args = parse_args()
```

Reads the command-line arguments.

---

## Run main

```python
main(args)
```

Starts the actual data preparation workflow.

---

## End separator

```python
print("*" * 60)
print("\n\n")
```

Adds another visual separator and blank lines to the logs.

---

# 22. Cell 7 — Create `train-model.py`

The notebook uses:

```python
%%writefile $script_folder/train-model.py
```

This creates:

```text
src/train-model.py
```

This script is responsible for model training and evaluation.

---

# 23. `train-model.py` — Imports

```python
import mlflow
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
```

---

## `mlflow`

```python
import mlflow
```

MLflow is used for experiment tracking.

The script calls:

```python
mlflow.autolog()
```

and:

```python
mlflow.log_param()
```

It also saves the trained model using:

```python
mlflow.sklearn.save_model()
```

---

## `glob`

```python
import glob
```

`glob` is used to find files matching a pattern.

The script searches for:

```text
*.csv
```

---

## `argparse`

Used for command-line arguments.

---

## pandas

Used to load and combine CSV data.

---

## NumPy

Used for the accuracy calculation.

---

## `train_test_split`

```python
from sklearn.model_selection import train_test_split
```

Used to divide data into training and testing subsets.

---

## `LogisticRegression`

```python
from sklearn.linear_model import LogisticRegression
```

Provides the classification model.

---

## `roc_auc_score`

```python
from sklearn.metrics import roc_auc_score
```

Calculates the Area Under the ROC Curve.

---

## `roc_curve`

```python
from sklearn.metrics import roc_curve
```

Calculates the points required to plot the ROC curve.

---

## Matplotlib

```python
import matplotlib.pyplot as plt
```

Used to create and save the ROC curve.

---

# 24. `train-model.py` — Main Function

```python
def main(args):
```

The function receives command-line arguments.

---

## Enable MLflow autologging

```python
mlflow.autolog()
```

This enables MLflow automatic logging.

MLflow can automatically record information generated during model training.

---

## Read training data

```python
df = get_data(args.training_data)
```

The training-data path comes from:

```python
args.training_data
```

It is passed to:

```python
get_data()
```

---

## Split data

```python
X_train, X_test, y_train, y_test = split_data(df)
```

The DataFrame is passed to:

```python
split_data()
```

The function returns four objects:

```text
X_train
X_test
y_train
y_test
```

---

## Train model

```python
model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
```

The model-training function receives:

- regularization rate
- training features
- test features
- training labels
- test labels

---

## Evaluate model

```python
eval_model(model, X_test, y_test)
```

The trained model is evaluated against the test data.

---

# 25. `get_data()` in `train-model.py`

```python
def get_data(data_path):

    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

    return df
```

---

## Find CSV files

```python
all_files = glob.glob(data_path + "/*.csv")
```

`glob.glob()` searches for files matching a pattern.

The pattern is:

```text
data_path/*.csv
```

This means:

> Find all CSV files inside the specified directory.

The results are stored in:

```python
all_files
```

---

## Read and combine CSV files

```python
df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
```

This performs two operations.

### Read each CSV

```python
pd.read_csv(f)
```

reads each file.

### Combine DataFrames

```python
pd.concat(...)
```

concatenates the DataFrames.

### `sort=False`

Prevents pandas from sorting the columns during concatenation.

---

## Return

```python
return df
```

Returns the combined DataFrame.

---

# 26. `split_data()`

```python
def split_data(df):
    print("Splitting data...")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test
```

---

## Print status

```python
print("Splitting data...")
```

Displays a progress message.

---

## Features

The feature columns are:

```text
Pregnancies
PlasmaGlucose
DiastolicBloodPressure
TricepsThickness
SerumInsulin
BMI
DiabetesPedigree
Age
```

They are selected with:

```python
df[[...]]
```

and converted to an array using:

```python
.values
```

The result is stored in:

```python
X
```

---

## Target

```python
df['Diabetic'].values
```

selects the target column:

```text
Diabetic
```

and converts it into an array.

The target is stored in:

```python
y
```

So:

```text
X = input features
y = target
```

---

# 27. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0
)
```

This divides the dataset into training and testing data.

---

## `X`

Feature matrix.

---

## `y`

Target values.

---

## `test_size=0.30`

30% of the data is reserved for testing.

Therefore approximately:

```text
70% → training
30% → testing
```

---

## `random_state=0`

Controls the random split.

Using the same value makes the split reproducible when the same data and environment are used.

---

# 28. `train_model()`

```python
def train_model(reg_rate, X_train, X_test, y_train, y_test):
```

This function trains the Logistic Regression model.

It receives:

- `reg_rate`
- `X_train`
- `X_test`
- `y_train`
- `y_test`

---

## Log the regularization parameter

```python
mlflow.log_param("Regularization rate", reg_rate)
```

MLflow records the regularization rate as a parameter.

The parameter name stored in MLflow is:

```text
Regularization rate
```

---

## Print training message

```python
print("Training model...")
```

Displays progress information.

---

## Create and train Logistic Regression

```python
model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
```

This line creates the model and immediately fits it.

---

## `LogisticRegression`

Creates a Logistic Regression classifier.

---

## `C=1/reg_rate`

This is an important relationship.

The notebook exposes:

```text
reg_rate
```

but scikit-learn's Logistic Regression uses:

```text
C
```

where the code calculates:

```python
C = 1 / reg_rate
```

For example:

| `reg_rate` | `C` |
|---:|---:|
| 0.1 | 10 |
| 0.01 | 100 |

---

## `solver="liblinear"`

Specifies the optimization algorithm used by Logistic Regression.

The notebook explicitly selects:

```text
liblinear
```

---

## `.fit(X_train, y_train)`

Trains the model using:

```text
X_train → features
y_train → labels
```

The trained model is assigned to:

```python
model
```

---

# 29. Save the Model

The script contains:

```python
mlflow.sklearn.save_model(model, args.model_output)
```

This saves the trained scikit-learn model to the path provided by:

```python
args.model_output
```

### Important source-based note

In the notebook exactly as supplied, `train_model()` refers to:

```python
args.model_output
```

even though `args` is not included as a parameter of `train_model()`.

The `main()` function passes:

```python
args.reg_rate
```

but calls:

```python
train_model(args.reg_rate, X_train, X_test, y_train, y_test)
```

without passing the complete `args` object.

Therefore, **the supplied script contains a variable-scope issue at this line** if `args` is not otherwise available globally.

The lab notebook itself does not provide a correction for this, so this guide preserves the code as supplied rather than silently changing it.

---

# 30. Return the Model

```python
return model
```

Returns the trained Logistic Regression model to `main()`.

---

# 31. `eval_model()`

```python
def eval_model(model, X_test, y_test):
```

This function evaluates the trained model.

It receives:

- trained model
- test features
- test labels

---

# 32. Calculate Accuracy

```python
y_hat = model.predict(X_test)
```

`predict()` produces class predictions for the test features.

The predictions are stored in:

```python
y_hat
```

---

## Compare predictions with actual labels

```python
y_hat == y_test
```

This creates Boolean values:

```text
True
False
True
...
```

A `True` means the prediction matches the actual label.

---

## Calculate average

```python
acc = np.average(y_hat == y_test)
```

`np.average()` calculates the average of the Boolean values.

Because:

```text
True ≈ 1
False ≈ 0
```

the average represents the proportion of correct predictions.

This gives the model's:

```text
Accuracy
```

---

## Print accuracy

```python
print('Accuracy:', acc)
```

Displays the accuracy.

---

# 33. Calculate AUC

```python
y_scores = model.predict_proba(X_test)
```

`predict_proba()` returns probability estimates for each class.

For binary classification, the result has two columns.

Conceptually:

```text
column 0 → probability of class 0
column 1 → probability of class 1
```

---

## Select positive-class probabilities

```python
y_scores[:,1]
```

The notation means:

```text
:  → all rows
1  → second column
```

Therefore:

```python
y_scores[:,1]
```

contains the predicted probability of the positive class.

---

## Calculate AUC

```python
auc = roc_auc_score(y_test,y_scores[:,1])
```

`roc_auc_score()` calculates the Area Under the ROC Curve.

Inputs:

```text
y_test
```

Actual labels.

```text
y_scores[:,1]
```

Predicted probabilities for the positive class.

---

## Print AUC

```python
print('AUC: ' + str(auc))
```

Converts the AUC value to a string and concatenates it with:

```text
AUC:
```

---

# 34. Generate ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
```

`roc_curve()` calculates points used to construct the ROC curve.

It returns:

### `fpr`

False Positive Rate.

### `tpr`

True Positive Rate.

### `thresholds`

Classification thresholds associated with the ROC points.

---

# 35. Create the Figure

```python
fig = plt.figure(figsize=(6, 4))
```

Creates a Matplotlib figure.

### `figsize=(6, 4)`

Specifies the figure dimensions.

Conceptually:

```text
width  = 6
height = 4
```

The values are in inches in Matplotlib's standard figure sizing.

---

# 36. Plot the 50% Reference Line

```python
plt.plot([0, 1], [0, 1], 'k--')
```

This draws the diagonal reference line.

The coordinates are:

```text
x = [0, 1]
y = [0, 1]
```

The style:

```text
k--
```

means:

- `k` → black
- `--` → dashed line

The notebook comments that this represents the diagonal 50% line.

---

# 37. Plot the Model ROC Curve

```python
plt.plot(fpr, tpr)
```

Plots:

```text
False Positive Rate
```

against:

```text
True Positive Rate
```

using the values calculated by `roc_curve()`.

---

# 38. Label the X-Axis

```python
plt.xlabel('False Positive Rate')
```

Labels the horizontal axis.

---

# 39. Label the Y-Axis

```python
plt.ylabel('True Positive Rate')
```

Labels the vertical axis.

---

# 40. Add the Title

```python
plt.title('ROC Curve')
```

Sets the plot title.

---

# 41. Save the ROC Curve

```python
plt.savefig("ROC-Curve.png")
```

Saves the figure as:

```text
ROC-Curve.png
```

This allows the plot to be stored as an output/artifact generated during execution.

---

# 42. `parse_args()` in `train-model.py`

```python
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)
    parser.add_argument("--model_output", dest='model_output',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args
```

---

## `--training_data`

```python
parser.add_argument("--training_data", dest='training_data',
                    type=str)
```

Defines the training-data input.

It becomes:

```python
args.training_data
```

---

## `--reg_rate`

```python
parser.add_argument("--reg_rate", dest='reg_rate',
                    type=float, default=0.01)
```

Defines the regularization-rate argument.

### `type=float`

The value is converted to a floating-point number.

### `default=0.01`

If no value is supplied, the default is:

```text
0.01
```

---

## `--model_output`

```python
parser.add_argument("--model_output", dest='model_output',
                    type=str)
```

Defines the output path for the model.

It becomes:

```python
args.model_output
```

---

# 43. Entry Point of `train-model.py`

```python
if __name__ == "__main__":
```

When the script is executed directly:

```python
print("\n\n")
print("*" * 60)

args = parse_args()

main(args)

print("*" * 60)
print("\n\n")
```

The script:

1. prints spacing
2. prints a separator
3. parses command-line arguments
4. executes `main(args)`
5. prints another separator

---

# 44. Define the Components

The notebook next explains the structure of an Azure ML component.

A component has three broad areas:

## Metadata

Examples:

```text
name
display_name
version
description
type
```

These describe and manage the component.

---

## Interface

The interface defines:

```text
inputs
outputs
```

For example, a training component receives training data and a regularization rate and produces a trained model.

---

## Command, code, and environment

These tell Azure ML:

- what command to execute
- where the source code is
- what software environment to use

---

# 45. Cell 8 — `prep-data.yml`

The notebook creates:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: prep_data
display_name: Prepare training data
version: 1
type: command
inputs:
  input_data: 
    type: uri_file
outputs:
  output_data:
    type: uri_folder
code: ./src
environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
command: >-
  python prep-data.py 
  --input_data ${{inputs.input_data}}
  --output_data ${{outputs.output_data}}
```

---

# 46. `$schema`

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
```

Specifies the Azure ML schema used to validate the component YAML.

The schema describes the expected structure and fields.

---

# 47. Component Name

```yaml
name: prep_data
```

The component's internal name is:

```text
prep_data
```

---

# 48. Display Name

```yaml
display_name: Prepare training data
```

This is the human-readable display name.

---

# 49. Version

```yaml
version: 1
```

Defines the component version.

---

# 50. Type

```yaml
type: command
```

Specifies that this is a command component.

A command component executes a command in an Azure ML environment.

---

# 51. Input Definition

```yaml
inputs:
  input_data: 
    type: uri_file
```

The component expects an input called:

```text
input_data
```

Its type is:

```text
uri_file
```

So Azure ML treats the input as a URI pointing to a file.

---

# 52. Output Definition

```yaml
outputs:
  output_data:
    type: uri_folder
```

The component produces:

```text
output_data
```

with type:

```text
uri_folder
```

This is appropriate because the script writes:

```text
diabetes.csv
```

inside the output directory.

---

# 53. Code Directory

```yaml
code: ./src
```

Tells Azure ML that the source code is located in:

```text
./src
```

The directory contains:

```text
prep-data.py
train-model.py
```

---

# 54. Environment

```yaml
environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
```

Specifies the Azure ML environment used to execute the component.

The notebook references:

```text
AzureML-sklearn-1.0-ubuntu20.04-py38-cpu
```

with:

```text
@latest
```

indicating the latest version of that named environment.

The environment provides the software needed to run the script.

---

# 55. Command

```yaml
command: >-
  python prep-data.py 
  --input_data ${{inputs.input_data}}
  --output_data ${{outputs.output_data}}
```

This is the command Azure ML executes.

Conceptually it runs:

```text
python prep-data.py
```

with two arguments.

---

## Input binding

```text
--input_data ${{inputs.input_data}}
```

The Azure ML component input:

```text
inputs.input_data
```

is connected to the script argument:

```text
--input_data
```

---

## Output binding

```text
--output_data ${{outputs.output_data}}
```

The Azure ML component output:

```text
outputs.output_data
```

is connected to:

```text
--output_data
```

---

# 56. Cell 9 — `train-model.yml`

The notebook creates:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: train_model
display_name: Train a logistic regression model
version: 1
type: command
inputs:
  training_data: 
    type: uri_folder
  reg_rate:
    type: number
    default: 0.01
outputs:
  model_output:
    type: mlflow_model
code: ./src
environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
command: >-
  python train-model.py 
  --training_data ${{inputs.training_data}} 
  --reg_rate ${{inputs.reg_rate}} 
  --model_output ${{outputs.model_output}}
```

---

# 57. Training Component Name

```yaml
name: train_model
```

Internal component name:

```text
train_model
```

---

# 58. Display Name

```yaml
display_name: Train a logistic regression model
```

Human-readable component name.

---

# 59. Version

```yaml
version: 1
```

Component version.

---

# 60. Component Type

```yaml
type: command
```

Defines a command component.

---

# 61. Training Data Input

```yaml
training_data: 
  type: uri_folder
```

The component expects training data as a URI folder.

This matches the output of the preparation component:

```yaml
output_data:
  type: uri_folder
```

This type compatibility allows the prepared data to be passed to the training step.

---

# 62. Regularization Input

```yaml
reg_rate:
  type: number
  default: 0.01
```

The component accepts:

```text
reg_rate
```

as a number.

Its default is:

```text
0.01
```

The Python script receives this through:

```text
--reg_rate
```

and eventually calculates:

```python
C = 1 / reg_rate
```

---

# 63. Model Output

```yaml
outputs:
  model_output:
    type: mlflow_model
```

The training component produces:

```text
model_output
```

with type:

```text
mlflow_model
```

This indicates that the output is represented as an MLflow model.

---

# 64. Code and Environment

```yaml
code: ./src
environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
```

The component uses:

```text
./src
```

for its code and the specified Azure ML scikit-learn environment for execution.

---

# 65. Training Command

```yaml
command: >-
  python train-model.py 
  --training_data ${{inputs.training_data}} 
  --reg_rate ${{inputs.reg_rate}} 
  --model_output ${{outputs.model_output}}
```

The command runs:

```text
python train-model.py
```

and passes:

```text
--training_data
--reg_rate
--model_output
```

The Azure ML expressions connect component-level inputs and outputs to the command-line arguments.

---

# 66. Load the Components

The notebook next loads the YAML component definitions.

```python
from azure.ai.ml import load_component
parent_dir = ""

prep_data = load_component(source=parent_dir + "./prep-data.yml")
train_logistic_regression = load_component(source=parent_dir + "./train-model.yml")
```

---

## Import

```python
from azure.ai.ml import load_component
```

Imports Azure ML's component-loading function.

---

## Parent directory

```python
parent_dir = ""
```

Sets the parent directory to an empty string.

This means the YAML files are referenced relative to the current location.

---

## Load preparation component

```python
prep_data = load_component(source=parent_dir + "./prep-data.yml")
```

Loads:

```text
prep-data.yml
```

into the Python object:

```python
prep_data
```

---

## Load training component

```python
train_logistic_regression = load_component(
    source=parent_dir + "./train-model.yml"
)
```

Loads:

```text
train-model.yml
```

into:

```python
train_logistic_regression
```

---

# 67. Build the Pipeline

The notebook explains that the pipeline should execute:

```text
prep_data
   |
   v
train_logistic_regression
```

The output of the first component becomes the input to the second component.

---

# 68. Import Pipeline Utilities

```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
```

---

## `Input`

Used to specify an Azure ML input.

---

## `AssetTypes`

Contains Azure ML asset-type constants.

The notebook uses:

```python
AssetTypes.URI_FILE
```

---

## `pipeline`

Imports the Azure ML pipeline decorator.

---

# 69. Define the Pipeline Function

```python
@pipeline()
def diabetes_classification(pipeline_job_input):
```

---

## `@pipeline()`

The decorator tells Azure ML that this function defines a pipeline.

---

## Function name

```python
diabetes_classification
```

This represents the complete workflow.

---

## Pipeline input

```python
pipeline_job_input
```

The pipeline expects one input.

That input will be the registered diabetes data asset.

---

# 70. First Pipeline Step

```python
clean_data = prep_data(input_data=pipeline_job_input)
```

This invokes the `prep_data` component.

The pipeline input:

```text
pipeline_job_input
```

is connected to:

```text
input_data
```

of the preparation component.

The resulting component node is stored in:

```python
clean_data
```

---

# 71. Connect the Two Components

```python
train_model = train_logistic_regression(
    training_data=clean_data.outputs.output_data
)
```

The training component receives:

```python
clean_data.outputs.output_data
```

This is the output produced by the preparation component.

Therefore the pipeline creates this dependency:

```text
prep_data.output_data
          |
          v
train_logistic_regression.training_data
```

This is one of the most important concepts in the lab.

---

# 72. Return Pipeline Outputs

```python
return {
    "pipeline_job_transformed_data": clean_data.outputs.output_data,
    "pipeline_job_trained_model": train_model.outputs.model_output,
}
```

The pipeline exposes two outputs.

### Output 1

```text
pipeline_job_transformed_data
```

This refers to:

```python
clean_data.outputs.output_data
```

---

### Output 2

```text
pipeline_job_trained_model
```

This refers to:

```python
train_model.outputs.model_output
```

---

# 73. Create the Pipeline Job

```python
pipeline_job = diabetes_classification(
    Input(
        type=AssetTypes.URI_FILE,
        path="azureml:diabetes-data:1"
    )
)
```

This invokes the pipeline with a registered Azure ML data asset.

---

## `Input()`

Creates an Azure ML input definition.

---

## `type=AssetTypes.URI_FILE`

Specifies that the input is a URI file.

The notebook uses:

```python
AssetTypes.URI_FILE
```

---

## `path`

```python
path="azureml:diabetes-data:1"
```

References the registered data asset:

```text
diabetes-data
```

version:

```text
1
```

So the pipeline receives that registered asset as its input.

---

# 74. Print the Pipeline Configuration

The notebook contains:

```python
print(pipeline_job)
```

This displays the pipeline job configuration.

It allows you to inspect the generated pipeline object before submission.

---

# 75. Change Pipeline Output Modes

The notebook contains:

```python
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"
```

These lines configure how the pipeline outputs are handled.

Both outputs are set to:

```text
upload
```

---

## Transformed data output

```python
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
```

Sets the output mode for the transformed data.

---

## Trained model output

```python
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"
```

Sets the output mode for the trained model.

---

# 76. Set Pipeline-Level Compute

```python
pipeline_job.settings.default_compute = "aml-cluster"
```

This specifies the default compute target for the pipeline.

The compute target name is:

```text
aml-cluster
```

The setting is applied at pipeline level.

---

# 77. Set Pipeline-Level Datastore

```python
pipeline_job.settings.default_datastore = "workspaceblobstore"
```

This sets the default datastore.

The datastore specified is:

```text
workspaceblobstore
```

The pipeline can use this datastore for applicable data/output storage behavior.

---

# 78. Print Configuration Again

```python
print(pipeline_job)
```

The notebook prints the pipeline again so that the updated settings can be reviewed.

---

# 79. Submit the Pipeline Job

The final code is:

```python
# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_diabetes"
)
pipeline_job
```

---

## `ml_client.jobs`

Provides access to Azure ML job operations.

---

## `create_or_update()`

```python
ml_client.jobs.create_or_update(...)
```

Submits the pipeline job to the Azure ML workspace.

The method can create a new job or update an existing job definition.

---

## First argument

```python
pipeline_job
```

This is the pipeline job definition created earlier.

---

## `experiment_name`

```python
experiment_name="pipeline_diabetes"
```

Associates the submitted job with the experiment:

```text
pipeline_diabetes
```

This helps organize and track related runs.

---

## Final expression

```python
pipeline_job
```

Displays the returned/submitted pipeline job object.

---

# 80. Complete End-to-End Flow

The entire notebook can be understood as:

```text
1. Install/check Azure ML SDK
             |
             v
2. Authenticate with Azure
             |
             v
3. Create MLClient
             |
             v
4. Create src/
             |
             +----------------------+
             |                      |
             v                      v
      prep-data.py            train-model.py
             |                      |
             v                      v
      prep-data.yml           train-model.yml
             |                      |
             +----------+-----------+
                        |
                        v
                 load_component()
                        |
                        v
                  Build pipeline
                        |
                        v
                diabetes_classification
                        |
                        v
                 prep_data step
                        |
                        v
             transformed data
                        |
                        v
            training model step
                        |
                        +----------+
                        |          |
                        v          v
                 transformed    MLflow
                    data         model
                        |
                        v
                 Configure job
                        |
                        v
                  Submit job
                        |
                        v
               Azure ML workspace
```

---

# 81. Important Concepts to Remember

## Component

A reusable pipeline step.

In this notebook:

```text
prep_data
train_model
```

are components.

---

## Pipeline

A collection of connected components.

Here:

```text
prep_data → train_logistic_regression
```

---

## Component YAML

Describes:

- metadata
- inputs
- outputs
- code
- environment
- command

---

## Pipeline Input

The pipeline receives:

```text
azureml:diabetes-data:1
```

---

## Component Input/Output Connection

The key connection is:

```python
clean_data.outputs.output_data
```

being passed into:

```python
training_data
```

of the training component.

---

# 82. Data Types Used in the YAML

| Type | Used for | Notebook location |
|---|---|---|
| `uri_file` | Input data file | `prep-data.yml` |
| `uri_folder` | Prepared data directory | `prep-data.yml` |
| `uri_folder` | Training data input | `train-model.yml` |
| `number` | Regularization rate | `train-model.yml` |
| `mlflow_model` | Trained model output | `train-model.yml` |

---

# 83. Important Parameters — Quick Reference

| Parameter | Value | Purpose |
|---|---|---|
| `test_size` | `0.30` | 30% test data |
| `random_state` | `0` | Reproducible split |
| `reg_rate` | `0.01` default | Logistic Regression regularization input |
| `C` | `1/reg_rate` | Logistic Regression parameter |
| `solver` | `"liblinear"` | Logistic Regression solver |
| `figsize` | `(6, 4)` | ROC figure size |
| `default_compute` | `"aml-cluster"` | Pipeline compute |
| `default_datastore` | `"workspaceblobstore"` | Pipeline datastore |
| `experiment_name` | `"pipeline_diabetes"` | Experiment organization |
| data asset | `azureml:diabetes-data:1` | Pipeline input |

---

# 84. Feature Columns

The training script uses these eight features:

```text
Pregnancies
PlasmaGlucose
DiastolicBloodPressure
TricepsThickness
SerumInsulin
BMI
DiabetesPedigree
Age
```

The target column is:

```text
Diabetic
```

The preparation script normalizes these seven columns:

```text
Pregnancies
PlasmaGlucose
DiastolicBloodPressure
TricepsThickness
SerumInsulin
BMI
DiabetesPedigree
```

The notebook's normalization code does not include `Age` in `num_cols`.

---

# 85. Important Difference Between the Two Scripts

### `prep-data.py`

Responsible for data preparation:

```text
read
 ↓
remove missing values
 ↓
normalize selected columns
 ↓
save diabetes.csv
```

### `train-model.py`

Responsible for model training and evaluation:

```text
read prepared CSV files
 ↓
select X and y
 ↓
train/test split
 ↓
Logistic Regression
 ↓
Accuracy
 ↓
AUC
 ↓
ROC curve
 ↓
save MLflow model
```

---

# 86. Important Source-Based Code Observation

The supplied notebook contains this line inside `train_model()`:

```python
mlflow.sklearn.save_model(model, args.model_output)
```

However, the function is defined as:

```python
def train_model(reg_rate, X_train, X_test, y_train, y_test):
```

There is no `args` parameter in this function.

The caller is:

```python
model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
```

Therefore, as written in the supplied notebook, `args.model_output` may not be available inside `train_model()` unless `args` exists through some other scope.

This guide intentionally records that issue rather than silently modifying the notebook.

---

# 87. Another Important Observation

The YAML correctly defines:

```yaml
--model_output ${{outputs.model_output}}
```

and the argument parser defines:

```python
parser.add_argument("--model_output", dest='model_output',
                    type=str)
```

So the intended design is clearly:

```text
YAML model_output
       |
       v
--model_output
       |
       v
args.model_output
       |
       v
mlflow.sklearn.save_model(...)
```

The supplied Python function's scope is the part that does not directly reflect this intended flow.

---

# 88. Why Use a Pipeline?

A pipeline lets the workflow be represented as connected steps rather than one large script.

For this notebook:

```text
Data preparation
      ↓
Model training
```

This makes the workflow easier to:

- organize
- execute
- monitor
- reuse
- connect through explicit inputs and outputs

---

# 89. Viva / Exam Questions

## Q1. What is an Azure ML pipeline?

A pipeline is a workflow composed of multiple connected machine-learning components.

---

## Q2. What is a component?

A component is a reusable step that describes code, inputs, outputs, environment, and the command used to execute the step.

---

## Q3. What are the two components in this notebook?

They are:

```text
prep_data
train_model
```

---

## Q4. What does `prep-data.py` do?

It:

1. reads data
2. removes missing values
3. normalizes selected numeric columns
4. writes `diabetes.csv`

---

## Q5. What does `train-model.py` do?

It:

1. loads CSV data
2. splits the data
3. trains Logistic Regression
4. calculates Accuracy
5. calculates AUC
6. creates an ROC curve
7. saves the model

---

## Q6. Why is `%%writefile` used?

It writes the contents of a notebook cell to a file.

For example:

```python
%%writefile $script_folder/prep-data.py
```

creates:

```text
src/prep-data.py
```

---

## Q7. What does `os.makedirs(..., exist_ok=True)` do?

It creates a directory and avoids an error if the directory already exists.

---

## Q8. What does `dropna()` do?

It removes rows containing missing values.

---

## Q9. What does `MinMaxScaler` do?

It scales selected features using Min-Max normalization.

The default output range is:

```text
0 to 1
```

---

## Q10. Why is `Age` not in `num_cols`?

The supplied `normalize_data()` code lists seven columns and does not include `Age`.

The guide does not infer a reason because the notebook does not state one.

---

## Q11. What does `glob.glob(data_path + "/*.csv")` do?

It finds CSV files inside the specified data directory.

---

## Q12. Why use `pd.concat()`?

To combine the DataFrames loaded from the matching CSV files.

---

## Q13. What does `test_size=0.30` mean?

30% of the data is used for testing.

---

## Q14. What does `random_state=0` do?

It makes the train/test split reproducible when the same data and conditions are used.

---

## Q15. What is the relationship between `reg_rate` and `C`?

The script calculates:

```python
C = 1 / reg_rate
```

---

## Q16. What does `mlflow.autolog()` do?

It enables MLflow automatic logging during model training.

---

## Q17. What does `mlflow.log_param()` do?

It explicitly logs a parameter.

Here:

```python
mlflow.log_param("Regularization rate", reg_rate)
```

records the regularization rate.

---

## Q18. What does `predict_proba()` return?

It returns probability estimates for each class.

For the binary classification in this notebook:

```python
y_scores[:,1]
```

selects the probability of the second/positive class.

---

## Q19. What is AUC?

AUC stands for Area Under the ROC Curve.

The notebook calculates it with:

```python
roc_auc_score(y_test, y_scores[:,1])
```

---

## Q20. What does `roc_curve()` return?

It returns:

```text
fpr
tpr
thresholds
```

which are used to construct the ROC curve.

---

## Q21. What does `uri_file` mean in the component?

It identifies an input as a URI pointing to a file.

---

## Q22. What does `uri_folder` mean?

It identifies an input or output as a URI pointing to a folder.

---

## Q23. What does `mlflow_model` represent?

It is the type specified for the trained model output in the training component YAML.

---

## Q24. What is the pipeline input?

The pipeline uses:

```python
Input(
    type=AssetTypes.URI_FILE,
    path="azureml:diabetes-data:1"
)
```

So the registered data asset is:

```text
diabetes-data
version 1
```

---

## Q25. How are the two pipeline steps connected?

With:

```python
clean_data.outputs.output_data
```

passed as:

```python
training_data
```

to the training component.

---

## Q26. What does `default_compute` specify?

It specifies the default compute target for the pipeline.

The notebook uses:

```text
aml-cluster
```

---

## Q27. What does `default_datastore` specify?

It specifies the default datastore:

```text
workspaceblobstore
```

---

## Q28. What does `create_or_update()` do?

It submits the pipeline job to the Azure ML workspace, creating or updating the job definition.

---

## Q29. What experiment name is used?

```text
pipeline_diabetes
```

---

# 90. One-Minute Revision

Remember this sequence:

```text
Authenticate
    ↓
Create MLClient
    ↓
Create src folder
    ↓
Create prep-data.py
    ↓
Create train-model.py
    ↓
Create prep-data.yml
    ↓
Create train-model.yml
    ↓
load_component()
    ↓
@pipeline()
    ↓
prep_data
    ↓
train_logistic_regression
    ↓
Configure outputs
    ↓
Set compute
    ↓
Set datastore
    ↓
create_or_update()
```

The central idea is:

> **Python scripts implement the work, YAML files define reusable Azure ML components, and the pipeline connects those components into a workflow.**

---

# 91. Final Lab Summary

This notebook demonstrates an Azure ML pipeline that processes diabetes data and trains a Logistic Regression classifier.

The data flow is:

```text
diabetes-data:1
       |
       v
prep-data.py
       |
       +--> remove missing values
       |
       +--> MinMax normalization
       |
       v
diabetes.csv
       |
       v
train-model.py
       |
       +--> train/test split
       |
       +--> Logistic Regression
       |
       +--> Accuracy
       |
       +--> AUC
       |
       +--> ROC Curve
       |
       v
MLflow model
```

The most important Azure ML concepts demonstrated are:

- authentication
- `MLClient`
- command components
- component YAML
- component inputs and outputs
- registered data assets
- pipeline composition
- pipeline output configuration
- pipeline-level compute
- pipeline-level datastore
- job submission
- MLflow model output

---

## Quick Command Reference

```python
pip show azure-ai-ml
```

```python
ml_client = MLClient.from_config(credential=credential)
```

```python
os.makedirs(script_folder, exist_ok=True)
```

```python
prep_data = load_component(source="./prep-data.yml")
```

```python
train_logistic_regression = load_component(
    source="./train-model.yml"
)
```

```python
@pipeline()
def diabetes_classification(pipeline_job_input):
    clean_data = prep_data(input_data=pipeline_job_input)
    train_model = train_logistic_regression(
        training_data=clean_data.outputs.output_data
    )
```

```python
pipeline_job.settings.default_compute = "aml-cluster"
```

```python
pipeline_job.settings.default_datastore = "workspaceblobstore"
```

```python
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="pipeline_diabetes"
)
```

---

**End of Lab Guide**
