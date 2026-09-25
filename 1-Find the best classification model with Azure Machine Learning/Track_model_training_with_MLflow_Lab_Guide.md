# Lab Guide: Track Model Training in Notebooks with MLflow

## 1. Lab Objective

This lab demonstrates how to use **MLflow in an Azure Machine Learning notebook** to track machine-learning model training.

The notebook covers:

- Connecting to an Azure Machine Learning workspace.
- Verifying that `azure-ai-ml` and `mlflow` are installed.
- Loading a diabetes dataset with pandas.
- Separating features and labels.
- Splitting data into training and test sets.
- Creating an MLflow experiment.
- Using MLflow **autologging**.
- Disabling autologging.
- Using **custom logging**.
- Logging parameters.
- Logging metrics.
- Comparing different hyperparameter values.
- Comparing different estimators.
- Creating and saving an ROC curve.
- Reviewing logged parameters and metrics in Azure Machine Learning Studio.

The notebook trains models to predict the `Diabetic` label.

---

# 2. What is MLflow?

**MLflow** is used in this lab to track machine-learning experiments.

When training a model, you may want to record:

- Which model was used.
- Which hyperparameters were used.
- How accurate the model was.
- Which other metrics were produced.
- Files or plots associated with the run.

Without experiment tracking, it can become difficult to remember which model produced which result.

MLflow provides a way to organize this information into **experiments** and **runs**.

A useful mental model is:

```text
MLflow Experiment
       │
       ├── Run 1
       │     ├── Parameters
       │     ├── Metrics
       │     └── Model / Artifacts
       │
       ├── Run 2
       │     ├── Parameters
       │     ├── Metrics
       │     └── Model / Artifacts
       │
       └── Run 3
             ├── Parameters
             ├── Metrics
             └── Model / Artifacts
```

---

# 3. Azure Machine Learning and MLflow

The notebook is intended to run on an **Azure Machine Learning compute instance**.

The notebook explains that MLflow is already installed and integrated in this environment, so no special MLflow server configuration is required.

The workflow is approximately:

```text
Notebook
   ↓
Azure ML Compute Instance
   ↓
MLflow
   ↓
MLflow Experiment
   ↓
MLflow Runs
   ↓
Azure ML Studio
```

---

# 4. Cell 1 — Check the Azure ML SDK

The notebook begins with:

```python
pip show azure-ai-ml
```

## What is `pip`?

`pip` is Python's package-management tool.

It can be used to install and inspect Python packages.

For example:

```bash
pip install azure-ai-ml
```

installs the Azure Machine Learning SDK.

## What does `pip show` do?

```bash
pip show azure-ai-ml
```

displays information about the installed `azure-ai-ml` package.

It can show information such as:

- Package name
- Installed version
- Package location
- Dependencies

The purpose of this cell is to verify that the required Azure ML SDK is installed.

If it is not installed, the notebook suggests:

```bash
pip install azure-ai-ml
```

---

# 5. Connect to the Azure ML Workspace

The notebook next imports authentication and Azure ML client classes:

```python
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
```

---

# 6. `DefaultAzureCredential`

```python
DefaultAzureCredential
```

comes from:

```python
azure.identity
```

It provides a general Azure authentication mechanism.

It attempts to find an appropriate credential available in the current environment.

This allows the notebook to authenticate without directly putting a username and password into the code.

---

# 7. `InteractiveBrowserCredential`

```python
InteractiveBrowserCredential
```

is another Azure authentication mechanism.

It can authenticate the user through an interactive browser login.

The notebook uses it as a fallback if the default credential does not work.

---

# 8. `MLClient`

```python
from azure.ai.ml import MLClient
```

imports the Azure Machine Learning client.

`MLClient` provides the Python interface used to communicate with the Azure ML workspace.

Conceptually:

```text
Python Notebook
       ↓
    MLClient
       ↓
Azure ML Workspace
       ↓
Jobs / Experiments / Resources
```

---

# 9. Authentication `try` / `except`

The notebook uses:

```python
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()
```

Let's examine every line.

---

## `try`

```python
try:
```

The `try` statement tells Python:

> Attempt to execute the following code.

If an exception occurs, Python moves to the `except` block.

---

## Create the default credential

```python
credential = DefaultAzureCredential()
```

This creates a credential object.

The object is stored in:

```python
credential
```

---

## Test the credential

```python
credential.get_token("https://management.azure.com/.default")
```

This calls the credential object's `get_token()` method.

### `get_token()`

The purpose is to request an Azure access token for the specified resource/scope.

The parameter is:

```python
"https://management.azure.com/.default"
```

In this notebook, the call is used to verify that authentication is working.

---

## `except`

```python
except Exception as ex:
```

If authentication fails inside the `try` block, Python enters this section.

### `Exception`

`Exception` represents a Python exception.

### `as ex`

The exception is assigned to:

```python
ex
```

The notebook does not otherwise use `ex`.

---

## Browser fallback

```python
credential = InteractiveBrowserCredential()
```

If the default credential does not work, the notebook uses browser-based authentication.

The overall logic is:

```text
Try DefaultAzureCredential
          ↓
       Successful?
       /         \
     YES          NO
      ↓            ↓
 Continue     Browser login
```

---

# 10. Cell 4 — Create the ML Client

The notebook uses:

```python
# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)
```

The comment:

```python
# Get a handle to workspace
```

explains the purpose of the next line.

---

## `MLClient.from_config()`

```python
MLClient.from_config(credential=credential)
```

creates an Azure ML client using workspace configuration.

### Parameter: `credential`

```python
credential=credential
```

tells the ML client which authentication mechanism to use.

The resulting client is stored in:

```python
ml_client
```

This object is later used to interact with Azure ML resources.

---

# 11. Configure MLflow

The notebook explains that because it is running on an Azure Machine Learning compute instance, MLflow does not need additional configuration.

However, the notebook verifies that MLflow is installed.

The code is:

```python
pip show mlflow
```

---

# 12. `pip show mlflow`

```python
pip show mlflow
```

asks pip to display information about the installed MLflow package.

It is used to verify that the `mlflow` library is available.

If MLflow is not installed, the notebook suggests:

```bash
pip install mlflow
```

---

# 13. Prepare the Data

The notebook uses a diabetes classification dataset.

The dataset is stored at:

```text
../data/diabetes-data/diabetes.csv
```

The first step is to load it.

---

# 14. Import pandas

The notebook uses:

```python
import pandas as pd
```

This imports the **pandas** library.

The alias:

```python
pd
```

is assigned to pandas.

Pandas is commonly used for working with tabular data.

For example:

```python
pd.read_csv(...)
```

reads a CSV file into a pandas DataFrame.

---

# 15. Print a Message

The notebook uses:

```python
print("Reading data...")
```

### `print()`

`print()` displays text in the notebook output.

The output will be:

```text
Reading data...
```

This is simply a progress message.

---

# 16. Read the CSV File

The notebook uses:

```python
df = pd.read_csv('../data/diabetes-data/diabetes.csv')
```

Let's break this down.

## `pd.read_csv()`

`read_csv()` is a pandas function used to read a CSV file.

The parameter is:

```python
'../data/diabetes-data/diabetes.csv'
```

This is the relative path to the dataset.

### Meaning of `../`

`..` means the parent directory.

So the notebook looks for:

```text
../data/diabetes-data/diabetes.csv
```

---

## Result

The loaded data is stored in:

```python
df
```

`df` is a pandas **DataFrame**.

Conceptually:

```text
CSV file
   ↓
pd.read_csv()
   ↓
DataFrame
   ↓
df
```

---

# 17. Display the First Rows

The notebook uses:

```python
df.head()
```

### `head()`

`head()` displays the first rows of a pandas DataFrame.

By default, pandas displays the first **5 rows**.

The purpose is to inspect the data after loading it.

---

# 18. Split Features and Label

The notebook prints:

```python
print("Splitting data...")
```

This is another progress message.

Then it executes:

```python
X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
```

This is an important line because it separates the dataset into:

- Features: `X`
- Label/target: `y`

---

# 19. The `X` Features

The notebook selects these columns:

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

The expression is:

```python
df[
    [
        'Pregnancies',
        'PlasmaGlucose',
        'DiastolicBloodPressure',
        'TricepsThickness',
        'SerumInsulin',
        'BMI',
        'DiabetesPedigree',
        'Age'
    ]
]
```

These are the input features used by the machine-learning model.

---

# 20. `.values`

After selecting the feature columns, the notebook uses:

```python
.values
```

For a pandas DataFrame, `.values` returns the underlying data as a NumPy array representation.

Therefore:

```python
df[[...]].values
```

produces the feature array used as `X`.

---

# 21. The `y` Label

The notebook uses:

```python
df['Diabetic'].values
```

This selects the:

```text
Diabetic
```

column.

That column is the target/label the models learn to predict.

So:

```text
X = input features
y = target label
```

The complete conceptual structure is:

```text
Dataset
   │
   ├── X
   │    ├── Pregnancies
   │    ├── PlasmaGlucose
   │    ├── DiastolicBloodPressure
   │    ├── TricepsThickness
   │    ├── SerumInsulin
   │    ├── BMI
   │    ├── DiabetesPedigree
   │    └── Age
   │
   └── y
        └── Diabetic
```

---

# 22. Import `train_test_split`

The notebook uses:

```python
from sklearn.model_selection import train_test_split
```

This imports the scikit-learn function:

```python
train_test_split
```

It is used to divide data into training and testing subsets.

---

# 23. Train/Test Split

The notebook executes:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0
)
```

Let's examine every parameter.

---

## `X`

```python
X
```

contains the feature data.

---

## `y`

```python
y
```

contains the target labels.

---

## `test_size=0.30`

```python
test_size=0.30
```

means that 30% of the available data is assigned to the test set.

Approximately:

```text
100% data
   │
   ├── 70% → Training
   │
   └── 30% → Testing
```

Therefore:

```text
X_train → 70% feature data
X_test  → 30% feature data

y_train → 70% labels
y_test  → 30% labels
```

---

## `random_state=0`

```python
random_state=0
```

sets the random seed used by the splitting process.

Using a fixed value makes the split reproducible.

If the same data and parameters are used again, the same split can be reproduced.

---

# 24. The Four Resulting Arrays

After `train_test_split()`:

### `X_train`

Training features.

```text
Features used to train the model
```

### `X_test`

Testing features.

```text
Features used to evaluate the trained model
```

### `y_train`

Training labels.

```text
Correct target values for the training examples
```

### `y_test`

Testing labels.

```text
Correct target values used to evaluate predictions
```

Conceptually:

```text
Original Data
     │
     ↓
train_test_split()
     │
 ┌───┴────────────────┐
 ↓                    ↓
Training             Testing
 ↓                    ↓
X_train              X_test
y_train              y_test
```

---

# 25. Create an MLflow Experiment

The notebook imports MLflow:

```python
import mlflow
```

Then creates the experiment name:

```python
experiment_name = "mlflow-experiment-diabetes"
```

Finally:

```python
mlflow.set_experiment(experiment_name)
```

---

# 26. `import mlflow`

```python
import mlflow
```

imports the MLflow library.

The notebook then uses the `mlflow` object to:

- create/select an experiment,
- start runs,
- enable autologging,
- log parameters,
- log metrics.

---

# 27. Experiment Name

```python
experiment_name = "mlflow-experiment-diabetes"
```

This creates a Python variable:

```python
experiment_name
```

containing:

```text
mlflow-experiment-diabetes
```

The experiment provides a logical container for related MLflow runs.

---

# 28. `mlflow.set_experiment()`

```python
mlflow.set_experiment(experiment_name)
```

sets the MLflow experiment that subsequent runs belong to.

The parameter is:

```python
experiment_name
```

which contains:

```text
mlflow-experiment-diabetes
```

Conceptually:

```text
MLflow Experiment
        │
        ├── Logistic Regression Run
        ├── Logistic Regression Run
        ├── Decision Tree Run
        └── Decision Tree + ROC Run
```

---

# 29. Train and Track a Model with Autologging

The notebook imports:

```python
from sklearn.linear_model import LogisticRegression
```

Then:

```python
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)
```

This section demonstrates MLflow **autologging**.

---

# 30. `LogisticRegression`

```python
from sklearn.linear_model import LogisticRegression
```

imports scikit-learn's logistic regression estimator.

Logistic regression is commonly used for classification problems.

In this notebook, it is used to predict:

```text
Diabetic
```

---

# 31. `mlflow.start_run()`

The notebook uses:

```python
with mlflow.start_run():
```

A **run** is one tracked execution of a model-training experiment.

The `with` statement creates a context.

Conceptually:

```text
Start MLflow Run
      ↓
Train Model
      ↓
Log Information
      ↓
End MLflow Run
```

Using the context manager means the run is automatically ended when the block finishes.

---

# 32. `mlflow.sklearn.autolog()`

```python
mlflow.sklearn.autolog()
```

enables automatic MLflow logging for scikit-learn.

This means MLflow can automatically capture information associated with supported scikit-learn training operations.

The notebook specifically points out that you don't need to manually calculate evaluation metrics for this autologged run because MLflow automatically creates and logs supported metrics.

---

# 33. Train Logistic Regression

The notebook uses:

```python
model = LogisticRegression(
    C=1/0.1,
    solver="liblinear"
).fit(X_train, y_train)
```

There are two important parts:

```python
LogisticRegression(...)
```

and:

```python
.fit(X_train, y_train)
```

---

# 34. `C=1/0.1`

The notebook specifies:

```python
C=1/0.1
```

Python evaluates:

```text
1 / 0.1 = 10
```

So the actual value supplied to `C` is:

```text
10
```

The notebook later describes `0.1` as a **regularization rate** and uses its reciprocal as the scikit-learn `C` value.

---

# 35. What is `C`?

For scikit-learn's logistic regression, `C` controls the inverse of regularization strength.

Conceptually:

```text
Higher C
   ↓
Less regularization

Lower C
   ↓
More regularization
```

The notebook expresses its chosen regularization rate through:

```python
C=1/0.1
```

which gives:

```text
C = 10
```

---

# 36. `solver="liblinear"`

The parameter:

```python
solver="liblinear"
```

selects the optimization algorithm used by the logistic regression estimator.

The notebook specifically chooses the `liblinear` solver.

---

# 37. `.fit(X_train, y_train)`

```python
.fit(X_train, y_train)
```

trains the logistic regression model.

### `X_train`

Provides the training features.

### `y_train`

Provides the corresponding training labels.

After training:

```python
model
```

contains the fitted model.

---

# 38. Disable Autologging

The notebook next explains that custom logging can be used alongside autologging, or instead of autologging.

It disables autologging with:

```python
mlflow.sklearn.autolog(disable=True)
```

---

# 39. `autolog(disable=True)`

The method is:

```python
mlflow.sklearn.autolog()
```

and the parameter:

```python
disable=True
```

tells MLflow to disable scikit-learn autologging.

Therefore, subsequent scikit-learn models will no longer automatically receive the autologging behavior demonstrated earlier.

This allows the notebook to demonstrate **custom logging**.

---

# 40. Custom Logging

The notebook then uses:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

with mlflow.start_run():
    model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)

    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    mlflow.log_param("regularization_rate", 0.1)
    mlflow.log_metric("Accuracy", acc)
```

This run manually logs one parameter and one metric.

---

# 41. Import NumPy

```python
import numpy as np
```

imports NumPy and assigns it the common alias:

```python
np
```

NumPy is used here to calculate the average of correct/incorrect prediction comparisons.

---

# 42. Start a New MLflow Run

```python
with mlflow.start_run():
```

starts a new MLflow run.

This run is separate from the previous autologging run.

---

# 43. Train the Logistic Regression Model

```python
model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)
```

This creates and trains another logistic regression model.

The configuration is:

```text
C = 1 / 0.1 = 10
solver = liblinear
```

---

# 44. Make Predictions

The notebook uses:

```python
y_hat = model.predict(X_test)
```

## `predict()`

`predict()` uses the trained model to generate predictions for the supplied input data.

The parameter:

```python
X_test
```

contains the test features.

The predictions are stored in:

```python
y_hat
```

Conceptually:

```text
X_test
   ↓
trained model
   ↓
y_hat
```

---

# 45. Calculate Accuracy

The notebook uses:

```python
acc = np.average(y_hat == y_test)
```

This line contains several operations.

---

## `y_hat == y_test`

This compares every prediction with its actual label.

For example:

```text
Predicted: [1, 0, 1, 1]
Actual:    [1, 1, 1, 0]

Comparison:
           [True, False, True, False]
```

Correct predictions produce `True`.

Incorrect predictions produce `False`.

---

## `np.average()`

```python
np.average(...)
```

calculates the average of those Boolean values.

In numerical operations:

```text
True  → 1
False → 0
```

Therefore:

```text
average(correct comparisons)
```

is the proportion of correct predictions.

That is the accuracy.

For example:

```text
[True, True, True, False]

Average = 3 / 4
        = 0.75
        = 75%
```

---

# 46. Log a Parameter

The notebook uses:

```python
mlflow.log_param("regularization_rate", 0.1)
```

## `log_param()`

`mlflow.log_param()` records a parameter associated with the current MLflow run.

It receives two important arguments:

```python
"name"
"value"
```

Here:

```python
"regularization_rate"
```

is the parameter name.

And:

```python
0.1
```

is its value.

So MLflow records:

```text
regularization_rate = 0.1
```

---

# 47. Log a Metric

The notebook uses:

```python
mlflow.log_metric("Accuracy", acc)
```

## `log_metric()`

`mlflow.log_metric()` records a numerical metric for the current run.

The first argument:

```python
"Accuracy"
```

is the metric name.

The second:

```python
acc
```

is the calculated accuracy.

So MLflow records something like:

```text
Accuracy = 0.XX
```

where `XX` depends on the actual model results.

---

# 48. Why Track Hyperparameters?

The notebook explains that tracking models is useful when comparing models trained with different hyperparameter values.

The first custom-logging model uses:

```text
regularization rate = 0.1
```

The next model changes this to:

```text
regularization rate = 0.01
```

The accuracy is also logged for both runs.

This allows the experiment results to be compared in Azure ML Studio.

---

# 49. Second Logistic Regression Experiment

The notebook runs:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

with mlflow.start_run():
    model = LogisticRegression(C=1/0.01, solver="liblinear").fit(X_train, y_train)

    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    mlflow.log_param("regularization_rate", 0.01)
    mlflow.log_metric("Accuracy", acc)
```

The structure is almost identical to the previous run.

The important difference is:

```python
C=1/0.01
```

---

# 50. Calculate the New `C`

Python evaluates:

```text
1 / 0.01 = 100
```

Therefore this model uses:

```text
C = 100
```

The custom parameter recorded in MLflow is:

```text
regularization_rate = 0.01
```

So the two runs represent:

| Run | Regularization rate | `C` |
|---|---:|---:|
| First custom run | 0.1 | 10 |
| Second custom run | 0.01 | 100 |

The notebook then tracks accuracy for each run so that the results can be compared.

---

# 51. Try a Different Estimator

The notebook next explains that all previous models used logistic regression.

It now trains a:

```text
DecisionTreeClassifier
```

The code is:

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

with mlflow.start_run():
    model = DecisionTreeClassifier().fit(X_train, y_train)

    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    mlflow.log_param("estimator", "DecisionTreeClassifier")
    mlflow.log_metric("Accuracy", acc)
```

---

# 52. `DecisionTreeClassifier`

```python
from sklearn.tree import DecisionTreeClassifier
```

imports scikit-learn's decision-tree classification estimator.

A decision tree makes predictions using a sequence of decision rules based on the input features.

The notebook uses the default constructor:

```python
DecisionTreeClassifier()
```

No explicit hyperparameters are provided in this cell.

---

# 53. Train the Decision Tree

```python
model = DecisionTreeClassifier().fit(X_train, y_train)
```

This does two things:

1. Creates the decision-tree classifier.
2. Trains it using the training data.

### `X_train`

Training features.

### `y_train`

Training labels.

The trained model is stored in:

```python
model
```

---

# 54. Predict with the Decision Tree

```python
y_hat = model.predict(X_test)
```

The trained decision tree predicts labels for the test features.

The predictions are stored in:

```python
y_hat
```

---

# 55. Calculate Decision Tree Accuracy

```python
acc = np.average(y_hat == y_test)
```

As before:

1. Compare predictions to actual labels.
2. Convert the Boolean results to numerical values.
3. Calculate their average.

This produces the accuracy.

---

# 56. Log the Estimator

The notebook uses:

```python
mlflow.log_param("estimator", "DecisionTreeClassifier")
```

This records:

```text
estimator = DecisionTreeClassifier
```

The parameter name is:

```text
estimator
```

The parameter value is:

```text
DecisionTreeClassifier
```

This allows the estimator type to be identified when reviewing the experiment.

---

# 57. Log the Accuracy

```python
mlflow.log_metric("Accuracy", acc)
```

This records the calculated accuracy for the decision-tree run.

Now the MLflow experiment can contain results from both:

```text
LogisticRegression
```

and:

```text
DecisionTreeClassifier
```

---

# 58. What is an MLflow Artifact?

The notebook then introduces an **artifact**.

An artifact can be a file associated with a run.

Examples can include:

- images,
- plots,
- model files,
- data files,
- other output files.

The notebook's example is an ROC curve saved as an image.

The intended conceptual workflow is:

```text
Model
  ↓
ROC calculation
  ↓
ROC plot
  ↓
Image file
  ↓
MLflow artifact
```

---

# 59. Import the Required Libraries

The final code cell begins with:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
```

Each import has a specific purpose.

---

## `DecisionTreeClassifier`

```python
from sklearn.tree import DecisionTreeClassifier
```

imports the decision-tree classification estimator.

---

## `roc_curve`

```python
from sklearn.metrics import roc_curve
```

imports the scikit-learn function used to calculate points for a **Receiver Operating Characteristic (ROC) curve**.

---

## `matplotlib.pyplot`

```python
import matplotlib.pyplot as plt
```

imports Matplotlib's plotting interface.

The alias:

```python
plt
```

is used for plotting.

---

## NumPy

```python
import numpy as np
```

imports NumPy using the alias:

```python
np
```

It is used for calculating accuracy.

---

# 60. Start the Final MLflow Run

```python
with mlflow.start_run():
```

starts another MLflow run.

Everything inside this block is associated with this run.

---

# 61. Train the Decision Tree

```python
model = DecisionTreeClassifier().fit(X_train, y_train)
```

Creates and trains the decision tree using:

```text
X_train → features
y_train → labels
```

---

# 62. Predict Test Labels

```python
y_hat = model.predict(X_test)
```

Generates predicted class labels for the test data.

---

# 63. Calculate Accuracy

```python
acc = np.average(y_hat == y_test)
```

Calculates the proportion of correct predictions.

---

# 64. Get Prediction Scores

The notebook uses:

```python
y_scores = model.predict_proba(X_test)
```

## `predict_proba()`

`predict_proba()` returns predicted probabilities for each class.

For a binary classification problem, the result has two probability columns.

Conceptually:

```text
             Class 0     Class 1
Sample 1       0.80        0.20
Sample 2       0.25        0.75
Sample 3       0.60        0.40
```

The notebook later uses:

```python
y_scores[:,1]
```

to select the probability associated with the second class.

---

# 65. `y_scores[:,1]`

The expression:

```python
y_scores[:,1]
```

means:

- `:` → select every row.
- `1` → select column index 1.

Therefore it selects the probability for the second class for every test observation.

These scores are supplied to `roc_curve()`.

---

# 66. Calculate the ROC Curve

The notebook uses:

```python
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
```

## `roc_curve()`

The scikit-learn function:

```python
roc_curve()
```

calculates points used to construct an ROC curve.

The parameters are:

### First parameter

```python
y_test
```

The true class labels.

### Second parameter

```python
y_scores[:,1]
```

The model's scores/probabilities for the positive class.

---

# 67. Results of `roc_curve()`

The function returns three arrays:

```python
fpr
tpr
thresholds
```

### `fpr`

False Positive Rate.

Conceptually:

```text
False Positive Rate =
False Positives /
(False Positives + True Negatives)
```

### `tpr`

True Positive Rate.

This is also commonly called recall or sensitivity:

```text
True Positive Rate =
True Positives /
(True Positives + False Negatives)
```

### `thresholds`

The decision thresholds used to calculate the different ROC points.

---

# 68. Create a Figure

The notebook uses:

```python
fig = plt.figure(figsize=(6, 4))
```

## `plt.figure()`

Creates a new Matplotlib figure.

### Parameter: `figsize`

```python
figsize=(6, 4)
```

specifies the figure dimensions.

The tuple represents:

```text
(width, height)
```

So:

```text
6 × 4
```

inches is requested.

The figure object is stored in:

```python
fig
```

---

# 69. Plot the 50% Reference Line

The notebook uses:

```python
plt.plot([0, 1], [0, 1], 'k--')
```

## `plt.plot()`

`plot()` creates a line plot.

The first list:

```python
[0, 1]
```

provides the x coordinates.

The second:

```python
[0, 1]
```

provides the y coordinates.

The style:

```python
'k--'
```

means:

- `k` → black
- `--` → dashed line

The result is the diagonal reference line from:

```text
(0, 0)
```

to:

```text
(1, 1)
```

---

# 70. Plot the Model ROC Curve

The notebook uses:

```python
plt.plot(fpr, tpr)
```

Here:

```text
x-axis = fpr
y-axis = tpr
```

So this plots the ROC curve generated by the decision-tree model.

---

# 71. Label the X Axis

```python
plt.xlabel('False Positive Rate')
```

## `xlabel()`

Sets the x-axis label to:

```text
False Positive Rate
```

---

# 72. Label the Y Axis

```python
plt.ylabel('True Positive Rate')
```

## `ylabel()`

Sets the y-axis label to:

```text
True Positive Rate
```

---

# 73. Set the Plot Title

```python
plt.title('ROC Curve')
```

## `title()`

Sets the chart title to:

```text
ROC Curve
```

---

# 74. Save the Plot

```python
plt.savefig("ROC-Curve.png")
```

## `savefig()`

`savefig()` saves the current Matplotlib figure to a file.

The filename is:

```text
ROC-Curve.png
```

The result is an image file in PNG format.

---

# 75. Important Note About the Final Cell

The notebook's text says:

> "Run the following cell to log a parameter, metric, and an artifact."

The code **does save the ROC plot to `ROC-Curve.png`**, but the shown code does **not** contain an explicit:

```python
mlflow.log_artifact("ROC-Curve.png")
```

call.

Therefore, based strictly on the notebook code, the plot is **created and saved as a local file**, but the displayed cell does not explicitly log that file as an MLflow artifact.

This distinction is important:

```text
plt.savefig("ROC-Curve.png")
        ↓
Creates/saves the file
```

whereas:

```python
mlflow.log_artifact("ROC-Curve.png")
```

would explicitly tell MLflow to associate the file with the current run.

The notebook's final cell does not show that second operation.

---

# 76. Log the Final Parameter

The final cell uses:

```python
mlflow.log_param("estimator", "DecisionTreeClassifier")
```

This records:

```text
estimator = DecisionTreeClassifier
```

for the current run.

---

# 77. Log the Final Metric

The final cell uses:

```python
mlflow.log_metric("Accuracy", acc)
```

This records the calculated accuracy.

So the final run explicitly logs:

```text
Parameter:
estimator = DecisionTreeClassifier

Metric:
Accuracy = calculated accuracy
```

and also creates:

```text
ROC-Curve.png
```

as a saved file.

---

# 78. Review Results in Azure ML Studio

The final notebook instruction is to review the model results on the **Jobs** page of Azure Machine Learning Studio.

The notebook specifies:

### Parameters

Look under:

```text
Overview → Params
```

### Metrics

Look under:

```text
Overview → Metrics
```

and:

```text
Metrics tab
```

This is where the tracked experiment information can be reviewed.

---

# 79. Complete MLflow Workflow

The notebook demonstrates the following progression:

```text
Load Dataset
      ↓
Split Features / Label
      ↓
Train/Test Split
      ↓
Create MLflow Experiment
      ↓
Start MLflow Run
      ↓
 ┌───────────────────────────────┐
 │ Autologging                   │
 │ Logistic Regression           │
 └───────────────────────────────┘
      ↓
Disable Autologging
      ↓
 ┌───────────────────────────────┐
 │ Custom Logging                │
 │ Parameter + Metric            │
 └───────────────────────────────┘
      ↓
Change Regularization
      ↓
Compare Runs
      ↓
Try Decision Tree
      ↓
Compare Estimators
      ↓
Create ROC Curve
      ↓
Save ROC Image
      ↓
Review Results in Azure ML Studio
```

---

# 80. Experiment vs Run

This distinction is important for a viva.

## Experiment

An experiment is a logical grouping of related MLflow runs.

The notebook creates:

```python
experiment_name = "mlflow-experiment-diabetes"
```

and selects it with:

```python
mlflow.set_experiment(experiment_name)
```

---

## Run

A run represents one execution/training attempt.

The notebook creates runs with:

```python
with mlflow.start_run():
```

For example:

```text
Experiment: mlflow-experiment-diabetes

    Run 1 → Logistic Regression, autologging
    Run 2 → Logistic Regression, rate = 0.1
    Run 3 → Logistic Regression, rate = 0.01
    Run 4 → Decision Tree
    Run 5 → Decision Tree + ROC plot
```

---

# 81. Parameters vs Metrics vs Artifacts

This is one of the most important concepts in the lab.

## Parameters

Parameters describe the configuration of a model/run.

Example:

```python
mlflow.log_param("regularization_rate", 0.1)
```

or:

```python
mlflow.log_param("estimator", "DecisionTreeClassifier")
```

Think:

```text
"What settings did I use?"
```

---

## Metrics

Metrics describe numerical results.

Example:

```python
mlflow.log_metric("Accuracy", acc)
```

Think:

```text
"How well did the model perform?"
```

---

## Artifacts

Artifacts are files associated with a run.

The notebook demonstrates creation of:

```text
ROC-Curve.png
```

Think:

```text
"What files/results did the run produce?"
```

Again, in the supplied code, the ROC image is saved but not explicitly passed to `mlflow.log_artifact()`.

---

# 82. Autologging vs Custom Logging

The notebook demonstrates both approaches.

## Autologging

```python
mlflow.sklearn.autolog()
```

MLflow automatically tracks supported information from scikit-learn training.

Advantages:

- Less code.
- Less manual tracking.
- Convenient for experiments.

---

## Custom logging

Example:

```python
mlflow.log_param("regularization_rate", 0.1)
mlflow.log_metric("Accuracy", acc)
```

You explicitly decide what to record.

Advantages:

- Full control over what is tracked.
- Useful for custom parameters and metrics.
- Useful when you only want to record selected information.

---

# 83. Why Compare Different Regularization Rates?

The notebook first uses:

```text
regularization_rate = 0.1
```

and then:

```text
regularization_rate = 0.01
```

The corresponding `C` values are:

```text
1 / 0.1  = 10
1 / 0.01 = 100
```

The accuracy for each run is tracked.

This allows you to compare how changing a hyperparameter affects model performance.

Conceptually:

```text
Regularization = 0.1
       ↓
Train model
       ↓
Calculate Accuracy
       ↓
MLflow Run


Regularization = 0.01
       ↓
Train model
       ↓
Calculate Accuracy
       ↓
MLflow Run
```

The results can then be viewed together.

---

# 84. Why Compare Different Estimators?

The notebook also changes the algorithm.

First:

```text
LogisticRegression
```

Then:

```text
DecisionTreeClassifier
```

The accuracy is logged for each run.

This allows the experiment to compare model configurations using a common metric.

---

# 85. Full Code Concept Map

```text
                    MLflow Experiment
                           │
                           │
              mlflow-experiment-diabetes
                           │
             ┌─────────────┼──────────────┐
             ↓             ↓              ↓
          Run 1          Run 2          Run 3
             │             │              │
       Logistic Reg.  Logistic Reg.  Decision Tree
       Autologging     Rate = 0.1     Accuracy
             │             │
             ↓             ↓
          Metrics       Metrics
                           │
                           ↓
                     Run 4
                           │
                    Logistic Reg.
                    Rate = 0.01
                           │
                           ↓
                        Accuracy
                           │
                           ↓
                     Run 5
                           │
                    Decision Tree
                           │
                       ROC Curve
                           │
                           ↓
                     ROC-Curve.png
```

---

# 86. Line-by-Line Quick Reference

| Code | Purpose |
|---|---|
| `pip show azure-ai-ml` | Checks the Azure ML SDK installation |
| `from azure.identity import ...` | Imports Azure authentication classes |
| `DefaultAzureCredential()` | Creates default Azure authentication |
| `credential.get_token(...)` | Tests whether authentication works |
| `InteractiveBrowserCredential()` | Provides browser-based authentication fallback |
| `MLClient.from_config(...)` | Creates the Azure ML workspace client |
| `pip show mlflow` | Checks whether MLflow is installed |
| `import pandas as pd` | Imports pandas |
| `pd.read_csv(...)` | Loads the CSV dataset |
| `df.head()` | Displays the first rows |
| `df[[...]]` | Selects feature columns |
| `df['Diabetic']` | Selects the target column |
| `.values` | Gets the underlying array values |
| `train_test_split()` | Splits data into train/test sets |
| `test_size=0.30` | Allocates 30% to testing |
| `random_state=0` | Makes the split reproducible |
| `import mlflow` | Imports MLflow |
| `mlflow.set_experiment()` | Selects/creates the experiment |
| `mlflow.start_run()` | Starts an MLflow run |
| `mlflow.sklearn.autolog()` | Enables scikit-learn autologging |
| `LogisticRegression()` | Creates logistic regression model |
| `C=1/0.1` | Sets `C` to 10 |
| `solver="liblinear"` | Selects the liblinear solver |
| `.fit(X_train, y_train)` | Trains the model |
| `autolog(disable=True)` | Disables scikit-learn autologging |
| `model.predict(X_test)` | Produces predictions |
| `np.average(y_hat == y_test)` | Calculates accuracy |
| `mlflow.log_param()` | Logs a parameter |
| `mlflow.log_metric()` | Logs a numerical metric |
| `DecisionTreeClassifier()` | Creates a decision-tree classifier |
| `model.predict_proba()` | Produces class probabilities |
| `roc_curve()` | Calculates ROC curve points |
| `plt.figure()` | Creates a plotting figure |
| `figsize=(6,4)` | Sets figure dimensions |
| `plt.plot()` | Plots lines |
| `plt.xlabel()` | Labels x-axis |
| `plt.ylabel()` | Labels y-axis |
| `plt.title()` | Sets chart title |
| `plt.savefig()` | Saves the plot as a file |

---

# 87. Important Viva / Exam Questions

## Q1. What is MLflow?

MLflow is an experiment-tracking platform used in this lab to record information about machine-learning runs, including parameters, metrics, and associated outputs.

---

## Q2. Why use MLflow?

To track and compare different model training runs.

It helps answer questions such as:

```text
Which model was trained?
Which hyperparameters were used?
What accuracy did it achieve?
Which experiment/run produced the result?
```

---

## Q3. What is an MLflow experiment?

An experiment is a logical grouping of related MLflow runs.

The notebook uses:

```python
experiment_name = "mlflow-experiment-diabetes"
mlflow.set_experiment(experiment_name)
```

---

## Q4. What is an MLflow run?

A run represents one model-training execution.

It is created using:

```python
with mlflow.start_run():
```

---

## Q5. What does autologging do?

```python
mlflow.sklearn.autolog()
```

enables automatic MLflow logging for supported scikit-learn operations.

---

## Q6. How do you disable scikit-learn autologging?

```python
mlflow.sklearn.autolog(disable=True)
```

---

## Q7. What is custom logging?

Custom logging means explicitly choosing what to record.

For example:

```python
mlflow.log_param("regularization_rate", 0.1)
mlflow.log_metric("Accuracy", acc)
```

---

## Q8. What is a parameter?

A parameter describes a model configuration or setting.

Example:

```python
mlflow.log_param("regularization_rate", 0.1)
```

---

## Q9. What is a metric?

A metric is a numerical measurement of model performance.

Example:

```python
mlflow.log_metric("Accuracy", acc)
```

---

## Q10. What is an artifact?

An artifact is a file associated with an MLflow run.

The notebook creates:

```text
ROC-Curve.png
```

However, the supplied final cell does not explicitly call:

```python
mlflow.log_artifact("ROC-Curve.png")
```

so the file is saved but not explicitly logged as an MLflow artifact in the shown code.

---

## Q11. What is the target column?

```text
Diabetic
```

It is extracted with:

```python
df['Diabetic'].values
```

---

## Q12. Which features are used?

The notebook uses:

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

---

## Q13. What percentage of data is used for testing?

```text
30%
```

because:

```python
test_size=0.30
```

---

## Q14. Why use `random_state=0`?

To make the train/test split reproducible.

---

## Q15. Which models are trained?

The notebook trains:

```text
LogisticRegression
DecisionTreeClassifier
```

---

## Q16. Which logistic regression solver is used?

```text
liblinear
```

because:

```python
solver="liblinear"
```

---

## Q17. What is the first regularization rate?

```text
0.1
```

with:

```python
C=1/0.1
```

so:

```text
C=10
```

---

## Q18. What is the second regularization rate?

```text
0.01
```

with:

```python
C=1/0.01
```

so:

```text
C=100
```

---

## Q19. How is accuracy calculated manually?

The notebook uses:

```python
acc = np.average(y_hat == y_test)
```

The comparison:

```python
y_hat == y_test
```

produces Boolean values representing correct and incorrect predictions.

`np.average()` calculates the proportion of correct predictions.

---

## Q20. What does `predict_proba()` do?

```python
model.predict_proba(X_test)
```

returns predicted probabilities for each class.

The notebook uses:

```python
y_scores[:,1]
```

to select the probability for the second class.

---

## Q21. What is an ROC curve?

An ROC curve plots the relationship between:

```text
False Positive Rate
```

and:

```text
True Positive Rate
```

across different classification thresholds.

The notebook obtains these values using:

```python
roc_curve(y_test, y_scores[:,1])
```

---

# 88. Important Observation About the Notebook

The final section says that the ROC curve should be logged as an artifact.

However, the actual code shown is:

```python
plt.savefig("ROC-Curve.png")

mlflow.log_param("estimator", "DecisionTreeClassifier")
mlflow.log_metric("Accuracy", acc)
```

There is no:

```python
mlflow.log_artifact("ROC-Curve.png")
```

Therefore, strictly following the supplied notebook:

```text
ROC-Curve.png
      ↓
saved to file
```

but the displayed code does not explicitly perform:

```text
ROC-Curve.png
      ↓
MLflow artifact
```

This is an important distinction to remember when explaining the lab.

---

# 89. Final Mental Model

For an exam or viva, remember the entire lab as:

```text
Install / Verify Azure ML SDK
          ↓
Authenticate with Azure
          ↓
Create MLClient
          ↓
Verify MLflow
          ↓
Load diabetes.csv
          ↓
Separate X and y
          ↓
Train/Test Split
          ↓
Create MLflow Experiment
          ↓
Start Run
          ↓
Enable Autologging
          ↓
Train Logistic Regression
          ↓
Disable Autologging
          ↓
Custom Logging
          ↓
Change Regularization
          ↓
Compare Results
          ↓
Train Decision Tree
          ↓
Compare Estimators
          ↓
Generate ROC Curve
          ↓
Save ROC-Curve.png
          ↓
Review Params and Metrics
in Azure ML Studio
```

## One-Sentence Summary

> This lab demonstrates how to use MLflow with Azure Machine Learning to track diabetes classification experiments, compare logistic-regression hyperparameters and a decision-tree estimator, manually log parameters and accuracy metrics, and generate an ROC-curve output for review in Azure ML Studio.
