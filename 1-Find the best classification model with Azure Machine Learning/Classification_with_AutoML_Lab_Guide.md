# Lab Guide: Classification with Automated Machine Learning

## 1. Lab Objective

The objective of this lab is to use **Azure Machine Learning AutoML** to automatically try different machine-learning models and preprocessing configurations for a **classification** problem.

The notebook uses a diabetes dataset and predicts the target column:

```text
Diabetic
```

The AutoML experiment is configured to:

- Use the Azure ML compute cluster `aml-cluster`
- Predict `Diabetic`
- Optimize for `accuracy`
- Perform 5-fold cross-validation
- Train up to 5 trials
- Allow a maximum total training time of 60 minutes
- Allow each individual trial up to 20 minutes
- Enable early termination
- Exclude `LogisticRegression`
- Enable model explainability
- Request ONNX-compatible models

---

# 2. What is Automated Machine Learning?

Normally, solving a classification problem manually might look like:

```text
Dataset
   ↓
Data preprocessing
   ↓
Train Logistic Regression
   ↓
Evaluate
   ↓
Train Random Forest
   ↓
Evaluate
   ↓
Train SVM
   ↓
Evaluate
   ↓
Tune parameters
   ↓
Compare models
```

AutoML automates much of this process.

Conceptually:

```text
Training Data
      ↓
     AutoML
      ↓
 ┌────┴─────────────────────┐
 ↓                          ↓
Preprocessing           Algorithms
 ↓                          ↓
Missing values          Model 1
Scaling                 Model 2
Encoding                Model 3
                         ...
      ↓
Cross-validation
      ↓
Model evaluation
      ↓
Best-performing configurations
```

The important point is that you are configuring the experiment rather than manually writing the complete model-training process.

---

# 3. Cell 1 — Check Azure ML SDK

The notebook checks the Azure ML package with:

```python
pip show azure-ai-ml
```

## What is `pip`?

`pip` is Python's package-management tool.

It is used to install and inspect Python packages.

For example:

```bash
pip install azure-ai-ml
```

installs the Azure Machine Learning SDK.

## What does `pip show` do?

```bash
pip show azure-ai-ml
```

asks pip to display information about the installed package.

It can show:

- Name
- Version
- Summary
- Location
- Dependencies

This cell is essentially checking:

> Is the Azure ML SDK installed, and which version is installed?

If the package is not installed, the notebook indicates that you can run:

```bash
pip install azure-ai-ml
```

### Important distinction

`pip show azure-ai-ml` is a notebook/shell command rather than ordinary Python syntax in a standard `.py` file.

---

# 4. Authentication and Azure ML Imports

The notebook imports:

```python
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
```

## Line 1

```python
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
```

This imports two authentication classes from Azure's identity library.

### `DefaultAzureCredential`

```python
DefaultAzureCredential
```

is an Azure authentication mechanism.

It attempts to find an appropriate credential automatically, depending on the environment.

The advantage is that the notebook does not need to contain a username or password.

### `InteractiveBrowserCredential`

```python
InteractiveBrowserCredential
```

provides an interactive browser-based login.

It is used as a fallback if the default authentication method does not work.

---

## Line 2

```python
from azure.ai.ml import MLClient
```

This imports:

```python
MLClient
```

`MLClient` is the main Azure Machine Learning client used by this notebook.

Conceptually:

```text
Python Notebook
       ↓
   MLClient
       ↓
Azure ML Workspace
       ↓
Jobs / Data / Compute / Models
```

---

# 5. Authentication `try` / `except` Block

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

## `try`

```python
try:
```

Python's `try` block means:

> Try to execute the following code.

If an error occurs, Python can move to the `except` block.

---

## Create the default credential

```python
credential = DefaultAzureCredential()
```

This creates an authentication object.

The object is stored in:

```python
credential
```

Conceptually:

```text
credential
   ↓
Azure authentication mechanism
```

---

## Test the credential

```python
credential.get_token("https://management.azure.com/.default")
```

This attempts to obtain an Azure access token.

### `get_token()`

`get_token()` is a method of the credential object.

Its parameter is:

```python
"https://management.azure.com/.default"
```

This identifies the Azure resource/scope for which a token is requested.

In this notebook, the call is used to test whether the credential can successfully authenticate.

---

## `except`

```python
except Exception as ex:
```

If something goes wrong in the `try` block, Python enters this block.

### `Exception`

`Exception` represents a Python exception.

### `as ex`

The exception is stored in:

```python
ex
```

The notebook does not use `ex` afterward, but naming it makes the exception available for debugging if needed.

---

## Browser authentication fallback

```python
credential = InteractiveBrowserCredential()
```

If `DefaultAzureCredential` cannot authenticate successfully, the notebook falls back to browser-based authentication.

Overall:

```text
Try automatic Azure authentication
          ↓
       Does it work?
       /          \
     YES           NO
      ↓             ↓
Continue       Browser login
```

---

# 6. Create the Azure ML Client

The notebook uses:

```python
ml_client = MLClient.from_config(credential=credential)
```

This creates the Azure ML client.

## `MLClient`

`MLClient` represents the interface used by the notebook to communicate with the Azure Machine Learning workspace.

## `from_config()`

```python
MLClient.from_config(...)
```

creates an `MLClient` using workspace configuration.

The notebook supplies:

```python
credential=credential
```

### Parameter: `credential`

This tells Azure ML to use the authentication mechanism stored in the `credential` variable.

The resulting object is stored in:

```python
ml_client
```

Later, the notebook uses it to submit the AutoML job:

```python
ml_client.jobs.create_or_update(...)
```

Conceptually:

```text
ml_client
    ↓
Azure ML workspace
    ↓
jobs
    ↓
submit AutoML experiment
```

---

# 7. Preparing the Data

The notebook uses an Azure ML data asset represented as an **MLTable**.

The training data asset is:

```text
diabetes-training
```

with version:

```text
1
```

The notebook then imports the required Azure ML data classes:

```python
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input
```

---

# 8. `AssetTypes`

```python
from azure.ai.ml.constants import AssetTypes
```

This imports Azure ML's predefined asset-type constants.

One of the constants used by the notebook is:

```python
AssetTypes.MLTABLE
```

This identifies the input as an Azure ML table asset.

---

# 9. `Input`

```python
from azure.ai.ml import Input
```

This imports the Azure ML `Input` class.

`Input` describes an input that will be provided to an Azure ML job.

It does not simply mean "load all the data into the current Python process." Instead, it defines the data input that the Azure ML job should use.

---

# 10. Create the Training Data Input

The notebook creates the input with:

```python
my_training_data_input = Input(
    type=AssetTypes.MLTABLE,
    path="azureml:diabetes-training:1"
)
```

Let's examine each part.

## `Input()`

`Input()` creates an Azure ML input specification.

The result is stored in:

```python
my_training_data_input
```

---

## Parameter: `type`

```python
type=AssetTypes.MLTABLE
```

This tells Azure ML what kind of input is being supplied.

The value:

```python
AssetTypes.MLTABLE
```

identifies the input as an MLTable.

---

## What is MLTable?

An MLTable is an Azure ML representation for tabular data.

It provides information about how tabular data should be accessed and interpreted by Azure ML jobs.

---

## Parameter: `path`

```python
path="azureml:diabetes-training:1"
```

This identifies the registered Azure ML data asset.

The general structure is:

```text
azureml:<asset-name>:<version>
```

Here:

```text
azureml:
    diabetes-training:
    1
```

means:

- Asset name = `diabetes-training`
- Version = `1`

The complete input specification is stored in:

```python
my_training_data_input
```

and later passed to:

```python
training_data=my_training_data_input
```

---

# 11. Import AutoML

The notebook imports AutoML with:

```python
from azure.ai.ml import automl
```

This imports the Azure ML AutoML functionality.

---

# 12. Configure the AutoML Classification Job

The most important configuration is:

```python
classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)
```

This creates an AutoML classification-job configuration.

---

# 13. `automl.classification()`

```python
automl.classification()
```

configures an Azure ML AutoML classification experiment.

Classification means that the model predicts a class/category.

In this notebook, the target column is:

```text
Diabetic
```

Conceptually:

```text
Patient information
       ↓
      AutoML
       ↓
Diabetic = ?
       ↓
Class prediction
```

---

# 14. Parameter: `compute`

```python
compute="aml-cluster"
```

This specifies the Azure ML compute resource on which the experiment will run.

The compute resource is:

```text
aml-cluster
```

Conceptually:

```text
Notebook
   ↓
Submit job
   ↓
Azure ML
   ↓
aml-cluster
   ↓
Train models
```

---

# 15. Parameter: `experiment_name`

```python
experiment_name="auto-ml-class-dev"
```

This gives the experiment a name:

```text
auto-ml-class-dev
```

Experiment names help organize related Azure ML jobs.

---

# 16. Parameter: `training_data`

```python
training_data=my_training_data_input
```

This tells AutoML which data to use.

The value comes from the input specification created earlier:

```python
my_training_data_input
```

The flow is:

```text
Input(...)
      ↓
my_training_data_input
      ↓
training_data=...
      ↓
AutoML
```

---

# 17. Parameter: `target_column_name`

```python
target_column_name="Diabetic"
```

This tells AutoML which column the model should learn to predict.

Here the target/label is:

```text
Diabetic
```

Conceptually, a dataset could look like:

| Age | Glucose | BMI | BloodPressure | Diabetic |
|---:|---:|---:|---:|---|
| 45 | 160 | 31 | 80 | Yes |
| 32 | 95 | 24 | 70 | No |

The other columns are used as input features, while `Diabetic` is the target.

---

# 18. Parameter: `primary_metric`

```python
primary_metric="accuracy"
```

This tells AutoML which metric to prioritize when comparing candidate models.

## Accuracy

Accuracy can be expressed as:

```text
Accuracy = Correct Predictions / Total Predictions
```

For example, if a model correctly predicts 90 out of 100 observations:

```text
Accuracy = 90 / 100
         = 0.90
         = 90%
```

The notebook specifically chooses **accuracy** as its primary metric.

---

# 19. Parameter: `n_cross_validations`

```python
n_cross_validations=5
```

This specifies **5-fold cross-validation**.

Conceptually:

```text
Dataset
   ↓
 ┌───┬───┬───┬───┬───┐
 │ 1 │ 2 │ 3 │ 4 │ 5 │
 └───┴───┴───┴───┴───┘
```

The data is divided into five folds.

The model is trained and evaluated repeatedly using different folds for validation.

For example:

```text
Round 1:
Train: 2 3 4 5
Test:  1

Round 2:
Train: 1 3 4 5
Test:  2

...

Round 5:
Train: 1 2 3 4
Test:  5
```

This provides a more robust estimate of model performance than relying on a single split.

---

# 20. Parameter: `enable_model_explainability`

```python
enable_model_explainability=True
```

This enables model explainability.

The purpose is to help answer questions such as:

> Which features contributed to the model's predictions?

For example, after training you may want to understand the relative contribution of input features to predictions.

The value:

```python
True
```

means the option is enabled.

---

# 21. Configure AutoML Limits

The notebook then uses:

```python
classification_job.set_limits(
    timeout_minutes=60,
    trial_timeout_minutes=20,
    max_trials=5,
    enable_early_termination=True,
)
```

This configures limits for the AutoML search.

---

# 22. Parameter: `timeout_minutes`

```python
timeout_minutes=60
```

This establishes a total experiment time limit of:

```text
60 minutes
```

It prevents the AutoML experiment from continuing indefinitely.

---

# 23. Parameter: `trial_timeout_minutes`

```python
trial_timeout_minutes=20
```

This sets a time limit for an individual AutoML trial.

A trial can be thought of as one candidate model/configuration attempt.

Conceptually:

```text
Overall experiment
        |
        +-- Trial 1
        +-- Trial 2
        +-- Trial 3
        +-- Trial 4
        +-- Trial 5
```

Each individual trial has a configured maximum time of 20 minutes.

---

# 24. Parameter: `max_trials`

```python
max_trials=5
```

This limits the number of trials to at most:

```text
5
```

Conceptually:

```text
AutoML
  ↓
Try candidate 1
Try candidate 2
Try candidate 3
Try candidate 4
Try candidate 5
  ↓
Compare results
```

---

# 25. Parameter: `enable_early_termination`

```python
enable_early_termination=True
```

This enables early termination for trials.

The idea is that an underperforming trial may be stopped before consuming all of its possible resources when AutoML determines that continuing it is not useful under the configured search strategy.

`True` means the feature is enabled.

---

# 26. Configure Training

The notebook then uses:

```python
classification_job.set_training(
    blocked_training_algorithms=["LogisticRegression"],
    enable_onnx_compatible_models=True
)
```

This configures additional training behavior.

---

# 27. Parameter: `blocked_training_algorithms`

```python
blocked_training_algorithms=["LogisticRegression"]
```

This tells AutoML not to use:

```text
LogisticRegression
```

as a candidate training algorithm.

The square brackets indicate a Python list:

```python
["LogisticRegression"]
```

The list currently contains one algorithm name.

Conceptually, a list can contain multiple entries:

```python
["AlgorithmA", "AlgorithmB"]
```

In this notebook, only `LogisticRegression` is blocked.

---

# 28. Parameter: `enable_onnx_compatible_models`

```python
enable_onnx_compatible_models=True
```

This requests ONNX-compatible models.

**ONNX** stands for:

> Open Neural Network Exchange

ONNX is a model representation designed to support model portability across supported environments and frameworks.

The notebook sets this option to:

```python
True
```

---

# 29. Complete AutoML Configuration

After:

```python
classification_job = automl.classification(...)
classification_job.set_limits(...)
classification_job.set_training(...)
```

the notebook has essentially created an experiment specification:

```text
                 AutoML Classification
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      Training        Target         Metric
       Data           Diabetic       Accuracy
          │
          ↓
    5-fold CV
          │
          ↓
    Model search
          │
     ┌────┴────┐
     ↓         ↓
   Models    Preprocessing
     │
     ↓
 Max 5 trials
     │
     ↓
  60-minute limit
     │
     ↓
 Compare performance
```

---

# 30. Submit the AutoML Job

The notebook then submits the job:

```python
returned_job = ml_client.jobs.create_or_update(
    classification_job
)
```

This is the point where the configured AutoML job is sent to Azure ML.

---

# 31. `ml_client.jobs`

`ml_client` is the Azure ML client.

It exposes different Azure ML resources and operations.

One of those is:

```python
ml_client.jobs
```

which provides functionality for working with jobs.

---

# 32. `create_or_update()`

The method:

```python
create_or_update()
```

creates or updates the specified job.

The argument is:

```python
classification_job
```

So the meaning is:

> Take this AutoML classification job configuration and create/update the corresponding Azure ML job.

---

# 33. `returned_job`

The result is stored in:

```python
returned_job
```

This object contains information about the submitted Azure ML job.

---

# 34. Get the Azure ML Studio URL

The notebook uses:

```python
aml_url = returned_job.studio_url
```

The job object contains:

```python
studio_url
```

which points to the Azure Machine Learning Studio page associated with the job.

That URL is stored in:

```python
aml_url
```

---

# 35. Print the Monitoring URL

Finally:

```python
print("Monitor your job at", aml_url)
```

## `print()`

`print()` displays information in the notebook output.

The output is conceptually:

```text
Monitor your job at <Azure ML Studio URL>
```

You can use that URL to monitor the AutoML experiment.

---

# 36. Complete Code Flow

The entire notebook can be understood as five major stages:

```text
1. Check SDK
      ↓
2. Authenticate
      ↓
3. Connect to Azure ML workspace
      ↓
4. Configure AutoML
      ↓
5. Submit and monitor job
```

More specifically:

```text
┌───────────────────────────────┐
│ Azure ML SDK                  │
│ pip show azure-ai-ml          │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ Authentication                │
│ DefaultAzureCredential        │
│ InteractiveBrowserCredential  │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ MLClient                      │
│ MLClient.from_config()        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ Data                          │
│ diabetes-training:1           │
│ MLTable                       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ AutoML Classification         │
│ Target = Diabetic             │
│ Metric = Accuracy             │
│ CV = 5                        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ Limits                        │
│ 60 min                        │
│ 20 min/trial                  │
│ 5 trials                      │
│ Early termination             │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ Training restrictions         │
│ Block Logistic Regression     │
│ ONNX compatible = True        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│ Submit                        │
│ ml_client.jobs.create_or_update│
└───────────────┬───────────────┘
                ↓
       Azure ML Studio
```

---

# 37. Line-by-Line Quick Reference

| Code | Meaning |
|---|---|
| `pip show azure-ai-ml` | Checks whether Azure ML SDK is installed |
| `from azure.identity import ...` | Imports Azure authentication classes |
| `from azure.ai.ml import MLClient` | Imports Azure ML client |
| `DefaultAzureCredential()` | Creates automatic/default Azure credential |
| `get_token(...)` | Tests whether authentication can obtain a token |
| `InteractiveBrowserCredential()` | Provides browser-based authentication |
| `MLClient.from_config(...)` | Creates Azure ML workspace client |
| `AssetTypes.MLTABLE` | Identifies input as MLTable |
| `Input(...)` | Creates Azure ML job input definition |
| `azureml:diabetes-training:1` | References registered data asset/version |
| `automl.classification()` | Creates AutoML classification configuration |
| `compute="aml-cluster"` | Specifies compute resource |
| `experiment_name=...` | Names the experiment |
| `training_data=...` | Specifies training dataset |
| `target_column_name="Diabetic"` | Specifies prediction target |
| `primary_metric="accuracy"` | Metric used to compare models |
| `n_cross_validations=5` | Uses 5-fold cross-validation |
| `enable_model_explainability=True` | Enables model explainability |
| `set_limits()` | Configures AutoML resource/search limits |
| `timeout_minutes=60` | Total experiment time limit |
| `trial_timeout_minutes=20` | Individual trial time limit |
| `max_trials=5` | Maximum number of trials |
| `enable_early_termination=True` | Allows early stopping of trials |
| `set_training()` | Configures training behavior |
| `blocked_training_algorithms=[...]` | Excludes specified algorithms |
| `enable_onnx_compatible_models=True` | Requests ONNX-compatible models |
| `jobs.create_or_update()` | Submits/creates the job |
| `studio_url` | Gets the Azure ML Studio job URL |
| `print()` | Displays the monitoring URL |

---

# 38. Important Viva / Exam Questions

## Q1. Is this supervised or unsupervised learning?

**Supervised learning**, because the dataset has a target column:

```text
Diabetic
```

The model learns from labeled examples.

---

## Q2. Is this classification or regression?

**Classification.**

The notebook explicitly uses:

```python
automl.classification()
```

---

## Q3. What is the target variable?

```text
Diabetic
```

because:

```python
target_column_name="Diabetic"
```

---

## Q4. What is the primary evaluation metric?

```text
Accuracy
```

because:

```python
primary_metric="accuracy"
```

---

## Q5. How many cross-validation folds are used?

```text
5
```

because:

```python
n_cross_validations=5
```

---

## Q6. How many trials can AutoML run?

At most:

```text
5
```

because:

```python
max_trials=5
```

---

## Q7. What algorithm is explicitly excluded?

```text
LogisticRegression
```

because:

```python
blocked_training_algorithms=["LogisticRegression"]
```

---

## Q8. What is the total timeout?

```text
60 minutes
```

because:

```python
timeout_minutes=60
```

---

## Q9. What is the timeout for an individual trial?

```text
20 minutes
```

because:

```python
trial_timeout_minutes=20
```

---

## Q10. Which compute resource is used?

```text
aml-cluster
```

because:

```python
compute="aml-cluster"
```

---

## Q11. What is `MLClient` used for?

It provides the Python interface for interacting with the Azure Machine Learning workspace and, in this notebook, is ultimately used to submit the AutoML job.

---

## Q12. Why is `InteractiveBrowserCredential` present?

It acts as a fallback if:

```python
DefaultAzureCredential()
```

cannot successfully authenticate.

---

## Q13. Why use `MLTable`?

The notebook's training data is represented as an Azure ML tabular data asset that AutoML can consume.

---

## Q14. Does the notebook manually train a model?

No.

The notebook **configures and submits an AutoML job**. Azure ML performs the model-search/training process on the specified compute.

---

# 39. Important Limitation of the Notebook

The notebook ends after submitting the AutoML job.

It does not contain code to explicitly:

- retrieve the best model,
- display the final model,
- calculate a separate test-set accuracy,
- download the model,
- register the selected model,
- make predictions on new data.

Instead, the final section gives the Azure ML Studio URL:

```python
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```

The next logical stage of a more complete lab would be:

1. Monitor the AutoML run.
2. Inspect the trials and metrics.
3. Identify the resulting/best model according to the configured metric.
4. Retrieve or register the model.
5. Evaluate it on appropriate held-out data.
6. Use it for predictions.

These steps are **not present in the uploaded notebook**, so they are not represented as existing notebook functionality in this guide.

---

# 40. Final Mental Model

For an exam or viva, remember the complete workflow as:

```text
DATA
 ↓
MLTable
 ↓
Azure authentication
 ↓
MLClient
 ↓
AutoML Classification
 ↓
Target = Diabetic
 ↓
Metric = Accuracy
 ↓
5-fold Cross Validation
 ↓
Maximum 5 trials
 ↓
60-minute experiment limit
 ↓
Block Logistic Regression
 ↓
Submit Job
 ↓
Monitor in Azure ML Studio
```

## One-Sentence Summary

> This lab uses Azure Machine Learning AutoML to configure and submit a classification experiment on the `diabetes-training` MLTable, predicting `Diabetic` with accuracy as the primary metric, 5-fold cross-validation, up to 5 trials, and specified training/resource constraints.
