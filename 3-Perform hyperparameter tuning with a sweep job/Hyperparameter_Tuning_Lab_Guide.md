# Hyperparameter Tuning with a Sweep Job — Detailed Azure ML Lab Guide

## 1. Lab Title

**Tune hyperparameters with a sweep job**

---

## 2. Objective

This notebook demonstrates how to use an **Azure Machine Learning sweep job** to tune a model hyperparameter.

The example uses a Logistic Regression model for diabetes classification.

The hyperparameter being tuned is:

```text
Regularization rate
```

The notebook tests three values:

```text
0.01
0.1
1
```

The objective is to compare the resulting model performance using:

```text
training_accuracy_score
```

The workflow is:

```text
Create training script
        ↓
Run one normal command job
        ↓
Verify the script works
        ↓
Define hyperparameter search space
        ↓
Create sweep job
        ↓
Run multiple trials
        ↓
Compare trial accuracy
```

---

# 3. What Is Hyperparameter Tuning?

A **hyperparameter** is a value that influences model training but is not learned directly from the training data.

For example, this notebook tunes:

```text
regularization rate
```

For other machine-learning algorithms, hyperparameters can include:

- learning rate
- batch size
- number of trees
- maximum tree depth
- number of hidden layers

The notebook explains that different hyperparameter values can affect:

- model performance
- training time

Therefore, multiple values may need to be tested to find a useful configuration.

---

# 4. What Is a Sweep Job?

An Azure ML **sweep job** runs multiple training trials using different hyperparameter values.

In this notebook, the search space is:

```text
0.01
0.1
1
```

The sweep uses:

```text
grid
```

sampling.

So Azure ML evaluates the specified combinations and allows the trials to be compared.

---

# 5. Overall Workflow

```text
Azure ML Workspace
        |
        v
   Training Script
        |
        v
   Command Job
        |
        | verify script
        v
 Hyperparameter Search Space
        |
        v
    Sweep Job
        |
        +----------+----------+
        |          |          |
        v          v          v
    Trial 1    Trial 2    Trial 3
    reg=0.01   reg=0.1    reg=1
        |          |          |
        +----------+----------+
                   |
                   v
        Compare Accuracy Scores
```

---

# 6. Notebook Structure

The notebook contains these major stages:

1. Check the Azure ML SDK
2. Authenticate with Azure
3. Connect to the workspace
4. Create the training script
5. Configure a command job
6. Submit the command job
7. Define the hyperparameter search space
8. Convert the command job into a sweep job
9. Configure sweep settings
10. Submit the sweep job
11. Review trial results

---

# 7. Cell 1 — Check Azure ML SDK

The notebook contains:

```python
pip show azure-ai-ml
```

## `pip show`

`pip show` displays information about an installed Python package.

The package being checked is:

```text
azure-ai-ml
```

This is the Azure Machine Learning Python SDK.

If the package is not installed, the notebook says to use:

```bash
pip install azure-ai-ml
```

---

# 8. Connect to the Azure ML Workspace

The notebook explains that the SDK needs to connect to an Azure ML workspace.

The workspace connection uses Azure authentication.

---

# 9. Cell 3 — Azure Authentication

```python
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()
```

---

## Import `DefaultAzureCredential`

```python
from azure.identity import DefaultAzureCredential
```

Imports the Azure identity credential class.

`DefaultAzureCredential` provides a standard way to authenticate with Azure.

---

## Import `InteractiveBrowserCredential`

```python
from azure.identity import InteractiveBrowserCredential
```

Provides browser-based authentication.

It is used as a fallback if the default credential cannot obtain a token.

---

## Import `MLClient`

```python
from azure.ai.ml import MLClient
```

`MLClient` is the Azure ML client used to communicate with the workspace and manage Azure ML resources.

---

# 10. `try` Block

```python
try:
```

The notebook first attempts to authenticate using:

```python
DefaultAzureCredential()
```

---

## Create the credential

```python
credential = DefaultAzureCredential()
```

Creates the default Azure credential object.

---

## Test authentication

```python
credential.get_token("https://management.azure.com/.default")
```

Attempts to obtain an Azure management access token.

The scope is:

```text
https://management.azure.com/.default
```

The notebook uses this call to verify that the credential can successfully authenticate.

---

# 11. Authentication Fallback

```python
except Exception as ex:
```

If the authentication attempt fails, the code enters the `except` block.

It then executes:

```python
credential = InteractiveBrowserCredential()
```

This allows authentication through a browser.

The variable:

```python
ex
```

contains the exception object, although the notebook does not otherwise use it.

---

# 12. Cell 4 — Create the ML Client

```python
# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)
```

---

## `MLClient.from_config()`

Creates an Azure ML client using the workspace configuration.

The credential is supplied through:

```python
credential=credential
```

The resulting client is stored in:

```python
ml_client
```

This client is later used to submit jobs.

---

# 13. Create the Training Script

The notebook explains that hyperparameter tuning requires a training script that accepts a hyperparameter as an input.

This training script accepts two command-line parameters:

```text
--training_data
--reg_rate
```

---

## `--training_data`

This expects a string representing the training data path.

The notebook eventually connects it to the registered data asset:

```text
azureml:diabetes-data:1
```

---

## `--reg_rate`

This expects a number.

The default value is:

```text
0.01
```

This is the parameter that will be tuned by the sweep job.

---

# 14. Cell 6 — Create the `src` Folder

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

Imports Python's operating-system utilities.

---

## Define folder name

```python
script_folder = 'src'
```

Creates a variable containing:

```text
src
```

---

## Create folder

```python
os.makedirs(script_folder, exist_ok=True)
```

Creates the directory.

### `exist_ok=True`

Prevents an error if the directory already exists.

---

## Print confirmation

```python
print(script_folder, 'folder created')
```

Prints:

```text
src folder created
```

---

# 15. Cell 7 — Create `train.py`

The notebook uses:

```python
%%writefile $script_folder/train.py
```

This is a Jupyter/IPython magic command.

It writes the contents of the cell to:

```text
src/train.py
```

This script performs:

```text
Read data
   ↓
Split data
   ↓
Train Logistic Regression
   ↓
Calculate Accuracy
   ↓
Calculate AUC
   ↓
Create ROC curve
   ↓
Log metrics and artifact with MLflow
```

---

# 16. `train.py` — Import Libraries

The script begins with:

```python
import mlflow
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

Used to log:

- hyperparameters
- metrics
- the ROC curve artifact

---

## `argparse`

```python
import argparse
```

Used to read command-line parameters such as:

```text
--training_data
--reg_rate
```

---

## pandas

```python
import pandas as pd
```

Used to read the CSV data.

---

## NumPy

```python
import numpy as np
```

Used for numerical operations, including the accuracy calculation.

---

## `train_test_split`

```python
from sklearn.model_selection import train_test_split
```

Splits the data into training and test sets.

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

Calculates the ROC AUC score.

---

## `roc_curve`

```python
from sklearn.metrics import roc_curve
```

Calculates the points required to create an ROC curve.

---

## Matplotlib

```python
import matplotlib.pyplot as plt
```

Used to create and save the ROC curve.

---

# 17. `main(args)`

```python
def main(args):
```

This function defines the main training workflow.

---

## Read data

```python
df = get_data(args.training_data)
```

Gets the path supplied through:

```python
args.training_data
```

and passes it to:

```python
get_data()
```

The resulting DataFrame is stored in:

```python
df
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

The function returns:

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

The regularization rate comes from:

```python
args.reg_rate
```

The model is trained using the training and test arrays.

---

## Evaluate model

```python
eval_model(model, X_test, y_test)
```

The trained model is evaluated on the test data.

---

# 18. `get_data()`

```python
def get_data(path):
    print("Reading data...")
    df = pd.read_csv(path)

    return df
```

---

## Function parameter

```python
path
```

Represents the location of the CSV data.

---

## Print status

```python
print("Reading data...")
```

Displays a progress message.

---

## Read CSV

```python
df = pd.read_csv(path)
```

Reads the CSV into a pandas DataFrame.

---

## Return

```python
return df
```

Returns the DataFrame.

---

# 19. `split_data()`

```python
def split_data(df):
    print("Splitting data...")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test
```

---

## Feature matrix

The script selects:

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

and stores them in:

```python
X
```

---

## Target

```python
df['Diabetic'].values
```

selects:

```text
Diabetic
```

as the target.

It is stored in:

```python
y
```

Therefore:

```text
X = features
y = target
```

---

## Train/test split

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0
)
```

### `test_size=0.30`

30% of the data is reserved for testing.

Approximately:

```text
70% → training
30% → testing
```

### `random_state=0`

Provides a reproducible split.

---

# 20. `train_model()`

```python
def train_model(reg_rate, X_train, X_test, y_train, y_test):
    mlflow.log_param("Regularization rate", reg_rate)
    print("Training model...")
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

    return model
```

---

## `reg_rate`

This is the hyperparameter being tuned.

The sweep will supply different values.

---

## Log the hyperparameter

```python
mlflow.log_param("Regularization rate", reg_rate)
```

Records the regularization rate in MLflow.

This is important because later the sweep results can be associated with the corresponding parameter value.

---

## Train Logistic Regression

```python
model = LogisticRegression(
    C=1/reg_rate,
    solver="liblinear"
).fit(X_train, y_train)
```

---

## Relationship between `reg_rate` and `C`

The script calculates:

```python
C = 1 / reg_rate
```

Therefore:

| Regularization rate | `C` |
|---:|---:|
| `0.01` | `100` |
| `0.1` | `10` |
| `1` | `1` |

---

## `solver="liblinear"`

Specifies the solver used by Logistic Regression.

---

## `.fit()`

```python
.fit(X_train, y_train)
```

Trains the model using the training features and labels.

---

# 21. `eval_model()`

```python
def eval_model(model, X_test, y_test):
```

Evaluates the trained model.

---

# 22. Accuracy

```python
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
mlflow.log_metric("training_accuracy_score", acc)
```

---

## Predictions

```python
y_hat = model.predict(X_test)
```

Produces predicted class labels.

---

## Compare predictions

```python
y_hat == y_test
```

Creates Boolean values representing whether each prediction is correct.

---

## Calculate accuracy

```python
acc = np.average(y_hat == y_test)
```

The average of the Boolean results represents the proportion of correct predictions.

---

## Print accuracy

```python
print('Accuracy:', acc)
```

Displays the score.

---

## Log accuracy

```python
mlflow.log_metric("training_accuracy_score", acc)
```

This is especially important for the sweep job.

The sweep configuration later specifies:

```python
primary_metric="training_accuracy_score"
```

Therefore, the metric name must match.

The relationship is:

```text
train.py
    |
    v
mlflow.log_metric(
    "training_accuracy_score",
    acc
)
    |
    v
sweep job reads
"training_accuracy_score"
```

---

# 23. AUC

```python
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
mlflow.log_metric("AUC", auc)
```

---

## Probability predictions

```python
y_scores = model.predict_proba(X_test)
```

Returns probability estimates for each class.

---

## Positive-class probability

```python
y_scores[:,1]
```

means:

```text
all rows
second column
```

So it selects the predicted probability of the positive class.

---

## AUC calculation

```python
auc = roc_auc_score(y_test, y_scores[:,1])
```

Calculates the ROC AUC score.

---

## Log AUC

```python
mlflow.log_metric("AUC", auc)
```

Records the AUC metric in MLflow.

---

# 24. ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
```

Returns:

```text
fpr
tpr
thresholds
```

where:

- `fpr` = False Positive Rate
- `tpr` = True Positive Rate
- `thresholds` = classification thresholds

---

## Create figure

```python
fig = plt.figure(figsize=(6, 4))
```

Creates a figure with size:

```text
6 × 4
```

---

## Reference line

```python
plt.plot([0, 1], [0, 1], 'k--')
```

Creates the diagonal reference line.

The style:

```text
k--
```

means black dashed line.

---

## Model ROC curve

```python
plt.plot(fpr, tpr)
```

Plots the model's ROC curve.

---

## Labels

```python
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
```

Adds the axis labels and title.

---

# 25. Save and Log ROC Artifact

```python
plt.savefig("ROC-Curve.png")
mlflow.log_artifact("ROC-Curve.png")
```

The first line saves the figure.

The second line uploads/logs the file as an MLflow artifact.

This is different from simply saving the file locally.

The notebook explicitly includes both operations.

---

# 26. `parse_args()`

```python
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args
```

---

## Create parser

```python
parser = argparse.ArgumentParser()
```

Creates a command-line argument parser.

---

## Training data parameter

```python
parser.add_argument(
    "--training_data",
    dest='training_data',
    type=str
)
```

Defines:

```text
--training_data
```

Its value becomes:

```python
args.training_data
```

and is treated as a string.

---

## Regularization parameter

```python
parser.add_argument(
    "--reg_rate",
    dest='reg_rate',
    type=float,
    default=0.01
)
```

Defines the hyperparameter.

### `type=float`

The value is converted to a floating-point number.

### `default=0.01`

If no value is provided, the default is:

```text
0.01
```

---

## Parse

```python
args = parser.parse_args()
```

Reads the command-line arguments.

---

## Return

```python
return args
```

Returns the argument object.

---

# 27. Script Entry Point

```python
if __name__ == "__main__":
```

The code inside this block runs when `train.py` is executed directly.

It:

```python
print("\n\n")
print("*" * 60)

args = parse_args()

main(args)

print("*" * 60)
print("\n\n")
```

The actual workflow is therefore:

```text
parse command-line arguments
            ↓
        main(args)
            ↓
        read data
            ↓
       split data
            ↓
      train model
            ↓
      evaluate model
```

---

# 28. Configure and Run a Command Job

The notebook next creates a normal Azure ML command job.

The notebook explicitly recommends testing the training script with a normal command job before performing hyperparameter tuning.

This is useful because it verifies that:

- the script can run
- the data asset can be accessed
- the environment works
- the model can train
- the metrics can be logged

Only after this basic job works does the notebook create the sweep.

---

# 29. Cell 9 — Import Command Job Utilities

```python
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes
```

---

## `command`

Used to create an Azure ML command job.

---

## `Input`

Used to define job inputs.

---

## `AssetTypes`

Provides Azure ML asset-type constants.

The notebook uses:

```python
AssetTypes.URI_FILE
```

---

# 30. Configure the Command Job

The notebook creates:

```python
job = command(
    code="./src",
    command="python train.py --training_data ${{inputs.diabetes_data}} --reg_rate ${{inputs.reg_rate}}",
    inputs={
        "diabetes_data": Input(
            type=AssetTypes.URI_FILE, 
            path="azureml:diabetes-data:1"
            ),
        "reg_rate": 0.01,
    },
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="aml-cluster",
    display_name="diabetes-train-mlflow",
    experiment_name="diabetes-training", 
    tags={"model_type": "LogisticRegression"}
    )
```

This creates the base command job that will later be converted into a sweep job.

---

# 31. `code="./src"`

```python
code="./src"
```

Specifies the directory containing the training script.

The directory contains:

```text
train.py
```

---

# 32. `command=...`

```python
command="python train.py --training_data ${{inputs.diabetes_data}} --reg_rate ${{inputs.reg_rate}}"
```

Specifies exactly what Azure ML should execute.

Conceptually:

```text
python train.py
```

with:

```text
--training_data
--reg_rate
```

---

## Training data binding

```text
${{inputs.diabetes_data}}
```

Connects the Azure ML job input named:

```text
diabetes_data
```

to the Python argument:

```text
--training_data
```

---

## Hyperparameter binding

```text
${{inputs.reg_rate}}
```

Connects the job input:

```text
reg_rate
```

to:

```text
--reg_rate
```

This connection is what later allows the sweep to replace `reg_rate` with different values.

---

# 33. Input: `diabetes_data`

```python
"diabetes_data": Input(
    type=AssetTypes.URI_FILE, 
    path="azureml:diabetes-data:1"
),
```

Defines the training-data input.

---

## `type=AssetTypes.URI_FILE`

The input is a URI pointing to a file.

---

## Registered data asset

```python
path="azureml:diabetes-data:1"
```

References:

```text
diabetes-data
```

version:

```text
1
```

---

# 34. Input: `reg_rate`

```python
"reg_rate": 0.01,
```

For the initial command job, the regularization rate is:

```text
0.01
```

This means the basic command job runs only one training configuration.

---

# 35. Environment

```python
environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest"
```

Specifies the Azure ML environment.

The environment provides the packages and runtime required to execute the script.

The notebook uses a scikit-learn environment based on:

```text
Ubuntu 20.04
Python 3.8
CPU
```

and requests:

```text
latest
```

version of the named environment.

---

# 36. Compute

```python
compute="aml-cluster"
```

Specifies the compute target:

```text
aml-cluster
```

The command job runs on this compute resource.

---

# 37. Display Name

```python
display_name="diabetes-train-mlflow"
```

Provides a human-readable name for the individual job.

---

# 38. Experiment Name

```python
experiment_name="diabetes-training"
```

Associates the command job with the experiment:

```text
diabetes-training
```

---

# 39. Tags

```python
tags={"model_type": "LogisticRegression"}
```

Adds metadata to the job.

The tag is:

```text
model_type = LogisticRegression
```

Tags can help identify or filter jobs.

---

# 40. Submit the Command Job

```python
returned_job = ml_client.create_or_update(job)
```

Submits the command job to Azure ML.

The returned job object is stored as:

```python
returned_job
```

---

# 41. Get Studio URL

```python
aml_url = returned_job.studio_url
```

Retrieves the Azure ML Studio URL associated with the job.

---

## Print monitoring URL

```python
print("Monitor your job at", aml_url)
```

Displays the URL so the job can be monitored in Azure ML Studio.

---

# 42. Why Run a Command Job First?

The notebook explicitly notes that the command job runs the training script once with:

```text
reg_rate = 0.01
```

Before performing hyperparameter tuning, it is a best practice in this notebook to verify that the script works as expected.

The logic is:

```text
First:
test one configuration

Then:
run multiple configurations
```

This helps avoid launching a large sweep if the underlying training script is broken.

---

# 43. Define the Search Space

After the command job completes successfully, the notebook creates a hyperparameter search space.

It wants to test:

```text
0.01
0.1
1
```

for:

```text
reg_rate
```

---

# 44. Cell 11 — Import `Choice`

```python
from azure.ai.ml.sweep import Choice
```

Imports the Azure ML sweep `Choice` class.

`Choice` represents a discrete set of possible hyperparameter values.

---

# 45. Create the Sweepable Command Job

```python
command_job_for_sweep = job(
    reg_rate=Choice(values=[0.01, 0.1, 1]),
)
```

The existing command job is called as a function.

The important change is:

```python
reg_rate=Choice(values=[0.01, 0.1, 1])
```

Instead of one fixed value:

```text
0.01
```

the sweep can now select from:

```text
0.01
0.1
1
```

---

# 46. `Choice`

```python
Choice(values=[0.01, 0.1, 1])
```

Defines a discrete search space.

The candidate values are:

```text
0.01
0.1
1
```

The sweep algorithm determines how those values are evaluated.

Because the later configuration uses:

```text
grid
```

the notebook is configuring a grid search over the supplied choices.

---

# 47. Configure the Sweep Job

The notebook uses:

```python
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="training_accuracy_score",
    goal="Maximize",
)
```

This converts the sweepable command job into a sweep job.

---

# 48. `compute`

```python
compute="aml-cluster"
```

Specifies where the sweep trials will run.

The compute target is:

```text
aml-cluster
```

---

# 49. `sampling_algorithm`

```python
sampling_algorithm="grid"
```

Specifies the search strategy.

The notebook states that allowed sampling algorithms include:

```text
random
grid
bayesian
```

The notebook chooses:

```text
grid
```

---

# 50. What Does Grid Search Mean Here?

The search space contains:

```text
0.01
0.1
1
```

A grid search evaluates the defined combinations in the search space.

Since this notebook has only one tuned parameter, the relevant candidate configurations are:

```text
Trial 1 → reg_rate = 0.01
Trial 2 → reg_rate = 0.1
Trial 3 → reg_rate = 1
```

The configured trial limit is slightly higher than the number of available values, as explained below.

---

# 51. `primary_metric`

```python
primary_metric="training_accuracy_score"
```

Specifies the metric used to compare trials.

This name must correspond to a metric logged by the training script.

The training script contains:

```python
mlflow.log_metric("training_accuracy_score", acc)
```

Therefore the names match exactly:

```text
Training script:
training_accuracy_score

Sweep configuration:
training_accuracy_score
```

This is critical.

If the sweep expects a metric name that the training script does not log, the sweep cannot use that metric as its primary optimization metric.

---

# 52. `goal`

```python
goal="Maximize"
```

Specifies that the sweep should seek a larger value of the primary metric.

The primary metric is:

```text
training_accuracy_score
```

Therefore the configured objective is to maximize accuracy.

The notebook's configuration uses the capitalized value:

```text
Maximize
```

---

# 53. Set Sweep Experiment Name

```python
sweep_job.experiment_name="sweep-diabetes"
```

Changes the experiment name for the sweep to:

```text
sweep-diabetes
```

This separates the sweep experiment from the earlier command-job experiment:

```text
diabetes-training
```

---

# 54. Set Sweep Limits

The notebook contains:

```python
sweep_job.set_limits(
    max_total_trials=4,
    max_concurrent_trials=2,
    timeout=7200
)
```

This controls how much the sweep is allowed to run.

---

# 55. `max_total_trials`

```python
max_total_trials=4
```

Sets the maximum number of trials to:

```text
4
```

The defined search space contains three distinct regularization-rate values:

```text
0.01
0.1
1
```

Therefore there are three explicit candidate values in this notebook, while the configured maximum total number of trials is four.

---

# 56. `max_concurrent_trials`

```python
max_concurrent_trials=2
```

Allows up to:

```text
2
```

trials to run concurrently.

Conceptually:

```text
Trial 1 ─┐
         ├─ running at the same time
Trial 2 ─┘

Trial 3
  ↓
after capacity becomes available
```

The actual scheduling is handled by Azure ML.

---

# 57. `timeout`

```python
timeout=7200
```

Specifies a time limit of:

```text
7200 seconds
```

That is:

```text
7200 / 60 = 120 minutes
```

So the configured sweep timeout is:

```text
2 hours
```

---

# 58. Submit the Sweep Job

The notebook contains:

```python
returned_sweep_job = ml_client.create_or_update(sweep_job)
```

This submits the sweep job to Azure ML.

---

## Returned job

The returned object is stored as:

```python
returned_sweep_job
```

---

# 59. Get the Studio URL

```python
aml_url = returned_sweep_job.studio_url
```

Retrieves the Azure ML Studio URL for the sweep job.

---

## Print monitoring link

```python
print("Monitor your job at", aml_url)
```

Displays the link where the sweep can be monitored.

---

# 60. Reviewing the Trials

The final notebook section says that when the job is complete, you can navigate to the job overview.

The:

```text
Trials
```

tab shows the models that were trained and how their:

```text
Accuracy
```

score differs for each regularization-rate value.

The important information to compare is:

```text
reg_rate
    vs.
training_accuracy_score
```

---

# 61. Complete Hyperparameter-Tuning Flow

The complete process is:

```text
1. Authenticate
       ↓
2. Connect to workspace
       ↓
3. Create train.py
       ↓
4. Define command job
       ↓
5. Use reg_rate = 0.01
       ↓
6. Submit command job
       ↓
7. Verify training works
       ↓
8. Define Choice([0.01, 0.1, 1])
       ↓
9. Create sweep job
       ↓
10. Use grid sampling
       ↓
11. Optimize training_accuracy_score
       ↓
12. Maximize accuracy
       ↓
13. Set trial limits
       ↓
14. Submit sweep
       ↓
15. Compare trials in Azure ML Studio
```

---

# 62. Key Relationship: Script → Command Job → Sweep

This is the most important concept in the notebook.

## Training script

The script accepts:

```text
--reg_rate
```

---

## Command job

The command job defines:

```python
"reg_rate": 0.01
```

and passes it to:

```text
--reg_rate
```

---

## Sweep job

The sweep replaces the single value with:

```python
Choice(values=[0.01, 0.1, 1])
```

Therefore:

```text
Single value
    ↓
0.01

becomes

Search space
    ↓
0.01
0.1
1
```

---

# 63. Key Relationship: Metric → Sweep Optimization

The training script logs:

```python
mlflow.log_metric("training_accuracy_score", acc)
```

The sweep configuration specifies:

```python
primary_metric="training_accuracy_score"
```

and:

```python
goal="Maximize"
```

Therefore:

```text
Model trial
    ↓
Calculate accuracy
    ↓
Log training_accuracy_score
    ↓
Azure ML reads metric
    ↓
Compare trials
    ↓
Maximize metric
```

---

# 64. Important Hyperparameter Relationship

The notebook uses:

```python
C=1/reg_rate
```

Therefore:

| `reg_rate` | Logistic Regression `C` |
|---:|---:|
| `0.01` | `100` |
| `0.1` | `10` |
| `1` | `1` |

The sweep changes `reg_rate`, and therefore indirectly changes `C`.

---

# 65. Important Metrics

The script records two metrics.

## Accuracy

```python
mlflow.log_metric("training_accuracy_score", acc)
```

This is the **primary sweep metric**.

---

## AUC

```python
mlflow.log_metric("AUC", auc)
```

AUC is also recorded, but it is **not** the metric selected for sweep optimization.

The sweep uses:

```text
training_accuracy_score
```

rather than:

```text
AUC
```

---

# 66. ROC Artifact

The script creates:

```text
ROC-Curve.png
```

using:

```python
plt.savefig("ROC-Curve.png")
```

and then logs it with:

```python
mlflow.log_artifact("ROC-Curve.png")
```

Therefore the ROC curve is both:

1. created as a file
2. logged as an MLflow artifact

---

# 67. Command Job vs Sweep Job

| Feature | Command Job | Sweep Job |
|---|---|---|
| Purpose | Test/run one configuration | Test multiple hyperparameter values |
| `reg_rate` | `0.01` | `0.01`, `0.1`, `1` |
| Sampling | None | `grid` |
| Primary metric | Not used for search | `training_accuracy_score` |
| Goal | Not applicable | `Maximize` |
| Trials | One configuration | Multiple trials |
| Experiment | `diabetes-training` | `sweep-diabetes` |

---

# 68. Important Parameters — Quick Reference

| Parameter | Value | Purpose |
|---|---|---|
| `test_size` | `0.30` | 30% test data |
| `random_state` | `0` | Reproducible split |
| `reg_rate` | `0.01` default | Regularization input |
| Search values | `0.01, 0.1, 1` | Hyperparameter candidates |
| `C` | `1/reg_rate` | Logistic Regression parameter |
| `solver` | `"liblinear"` | Logistic Regression solver |
| `sampling_algorithm` | `"grid"` | Sweep search strategy |
| `primary_metric` | `"training_accuracy_score"` | Metric used for optimization |
| `goal` | `"Maximize"` | Optimize toward higher metric |
| `max_total_trials` | `4` | Maximum total trials |
| `max_concurrent_trials` | `2` | Maximum simultaneous trials |
| `timeout` | `7200` | Sweep time limit in seconds |
| `compute` | `"aml-cluster"` | Compute target |
| command experiment | `"diabetes-training"` | Initial command-job experiment |
| sweep experiment | `"sweep-diabetes"` | Sweep experiment |
| data asset | `"azureml:diabetes-data:1"` | Training data |

---

# 69. Data Features

The model uses:

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

Target:

```text
Diabetic
```

---

# 70. Viva / Exam Questions

## Q1. What is hyperparameter tuning?

It is the process of trying different hyperparameter values to determine which configuration produces the desired model performance.

---

## Q2. What hyperparameter is tuned in this notebook?

```text
reg_rate
```

the regularization rate.

---

## Q3. What values are tested?

```text
0.01
0.1
1
```

---

## Q4. What is a sweep job?

A sweep job runs multiple trials with different hyperparameter configurations and compares them using a specified primary metric.

---

## Q5. Why run a command job before a sweep?

The notebook recommends testing the training script with a single command job first, so that script and environment problems can be identified before running multiple trials.

---

## Q6. What sampling algorithm is used?

```text
grid
```

---

## Q7. What other sampling algorithms does the notebook mention?

```text
random
grid
bayesian
```

---

## Q8. What is the primary metric?

```text
training_accuracy_score
```

---

## Q9. Where is the primary metric logged?

In `train.py`:

```python
mlflow.log_metric("training_accuracy_score", acc)
```

---

## Q10. What is the sweep goal?

```text
Maximize
```

---

## Q11. What is `Choice` used for?

It defines a discrete set of candidate hyperparameter values.

Example:

```python
Choice(values=[0.01, 0.1, 1])
```

---

## Q12. What does `max_total_trials=4` mean?

It sets the maximum total number of sweep trials to four.

---

## Q13. What does `max_concurrent_trials=2` mean?

At most two trials can run concurrently according to the sweep configuration.

---

## Q14. What does `timeout=7200` mean?

The sweep is configured with a maximum duration of 7200 seconds, or two hours.

---

## Q15. What is the relationship between `reg_rate` and `C`?

```python
C = 1 / reg_rate
```

---

## Q16. Which metric is logged but not optimized?

```text
AUC
```

The sweep's primary metric is:

```text
training_accuracy_score
```

---

## Q17. What does `mlflow.log_artifact()` do?

It logs a file as an MLflow artifact.

Here:

```python
mlflow.log_artifact("ROC-Curve.png")
```

logs the generated ROC curve.

---

## Q18. What is the training data asset?

```text
azureml:diabetes-data:1
```

---

## Q19. What compute target is used?

```text
aml-cluster
```

---

## Q20. What environment is used?

```text
AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
```

---

# 71. One-Minute Revision

Remember:

```text
SDK
 ↓
Authentication
 ↓
MLClient
 ↓
Create train.py
 ↓
Command job
 ↓
Test reg_rate = 0.01
 ↓
Choice([0.01, 0.1, 1])
 ↓
Sweep
 ↓
Grid search
 ↓
Primary metric = training_accuracy_score
 ↓
Goal = Maximize
 ↓
Submit
 ↓
Compare Trials
```

---

# 72. Final Summary

This notebook demonstrates **Azure ML hyperparameter tuning using a sweep job**.

The model is:

```text
Logistic Regression
```

The hyperparameter is:

```text
Regularization rate
```

The candidate values are:

```text
0.01
0.1
1
```

The sweep uses:

```text
Grid sampling
```

The optimization metric is:

```text
training_accuracy_score
```

The goal is:

```text
Maximize
```

The pipeline of experimentation is:

```text
Training Script
      ↓
Command Job
      ↓
Verify One Configuration
      ↓
Sweep Search Space
      ↓
Multiple Trials
      ↓
Compare Accuracy
```

The most important technical connection is:

```python
mlflow.log_metric("training_accuracy_score", acc)
```

matching:

```python
primary_metric="training_accuracy_score"
```

This allows Azure ML to use the metric reported by each training trial when comparing the sweep results.

---

## Final Quick Reference

### Authentication

```python
credential = DefaultAzureCredential()
```

### Workspace

```python
ml_client = MLClient.from_config(credential=credential)
```

### Training script

```text
src/train.py
```

### Command job

```python
job = command(...)
```

### Hyperparameter search space

```python
Choice(values=[0.01, 0.1, 1])
```

### Sweep

```python
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="training_accuracy_score",
    goal="Maximize",
)
```

### Limits

```python
sweep_job.set_limits(
    max_total_trials=4,
    max_concurrent_trials=2,
    timeout=7200
)
```

### Submit

```python
returned_sweep_job = ml_client.create_or_update(sweep_job)
```

### Monitor

```python
aml_url = returned_sweep_job.studio_url
```

---

**End of Lab Guide**
