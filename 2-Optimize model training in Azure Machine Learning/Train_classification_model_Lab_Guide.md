# Lab Guide: Train a Diabetes Classification Model

## 1. Lab Objective

This notebook reads a CSV file containing diabetes-related patient data and trains a **classification model** to predict whether a patient is diabetic.

The notebook's own introduction states that:

- The data is already preprocessed.
- No feature engineering is required.
- The evaluation methods were used during experimentation to decide whether the model was accurate enough.
- The notebook notes a preference for using MLflow autologging later to make model deployment easier.

The overall workflow is:

```text
Read CSV data
     ↓
Inspect data
     ↓
Separate features and target
     ↓
Split into training and test sets
     ↓
Train Logistic Regression
     ↓
Calculate Accuracy
     ↓
Calculate AUC
     ↓
Plot ROC Curve
```

---

# 2. Important Machine-Learning Concepts Used

Before going through the code, understand these terms:

### Features

Features are the input variables used by the model to make predictions.

This notebook uses:

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

### Target / Label

The target is the value the model tries to predict:

```text
Diabetic
```

### Classification

Classification means predicting a category/class rather than a continuous numerical value.

This notebook performs a diabetes classification task.

### Training data

The training data is used to learn the model parameters.

### Test data

The test data is kept separate from training and is used to evaluate the trained model.

### Accuracy

Accuracy measures the proportion of predictions that are correct.

### AUC

AUC stands for **Area Under the ROC Curve**.

It summarizes the model's ability to distinguish between the two classes across different classification thresholds.

### ROC Curve

A Receiver Operating Characteristic (ROC) curve plots:

```text
True Positive Rate
        vs.
False Positive Rate
```

for different classification thresholds.

---

# 3. Cell 0 — Notebook Introduction

The first cell is Markdown:

```markdown
# Train diabetes classification model

This notebook reads a CSV file and trains a model to predict diabetes in patients. The data is already preprocessed and requires no feature engineering.

The evaluation methods were used during experimentation to decide whether the model was accurate enough. Moving forward, there's a preference to use the autolog feature of MLflow to more easily deploy the model later on.
```

## Heading

```markdown
# Train diabetes classification model
```

The `#` creates a level-1 Markdown heading.

The heading tells us the purpose of the notebook:

> Train a classification model for diabetes prediction.

---

## Reading the CSV

The notebook states that it reads a CSV file.

The actual code later confirms the file:

```text
../data/diabetes-data/diabetes.csv
```

---

## Data is already preprocessed

The notebook explicitly states:

> The data is already preprocessed and requires no feature engineering.

This is important.

The notebook therefore does **not** contain additional preprocessing such as:

- scaling,
- encoding,
- feature creation,
- missing-value imputation,
- feature selection.

The guide should not assume that those operations occurred because they are not present in the notebook.

---

## Evaluation during experimentation

The notebook states that the evaluation methods were used during experimentation to determine whether the model was accurate enough.

The actual evaluation methods included later are:

```text
Accuracy
AUC
ROC Curve
```

---

## MLflow note

The notebook also mentions a preference to use MLflow's autolog feature in future work to make deployment easier.

This notebook itself, however, does **not** contain MLflow code.

That distinction is important:

```text
This notebook:
Model training + evaluation

MLflow:
Mentioned as a future/preferred workflow
but not implemented in the supplied code
```

---

# 4. Cell 1 — Read Data from Local File

The notebook contains the Markdown heading:

```markdown
## Read data from local file
```

The `##` creates a level-2 heading.

It tells us that the next section loads the dataset from a local file.

---

# 5. Cell 2 — Import pandas

The first line is:

```python
import pandas as pd
```

## `import`

`import` loads a Python library so that its functionality can be used in the notebook.

---

## `pandas`

`pandas` is a Python library commonly used for working with tabular data.

It provides objects such as:

```text
DataFrame
Series
```

---

## `as pd`

```python
import pandas as pd
```

assigns the short alias:

```text
pd
```

to pandas.

Therefore, instead of writing:

```python
pandas.read_csv(...)
```

the notebook can write:

```python
pd.read_csv(...)
```

---

# 6. Print a Progress Message

The notebook uses:

```python
print("Reading data...")
```

## `print()`

`print()` displays text in the notebook output.

The string:

```text
"Reading data..."
```

is simply a progress message.

The output will be:

```text
Reading data...
```

This line does not modify the dataset.

---

# 7. Read the CSV File

The notebook uses:

```python
df = pd.read_csv('../data/diabetes-data/diabetes.csv')
```

This is the main data-loading line.

---

## `pd.read_csv()`

`read_csv()` is a pandas function for reading a CSV file.

The function receives this parameter:

```python
'../data/diabetes-data/diabetes.csv'
```

This is the path to the CSV file.

---

## Understanding the path

```text
../data/diabetes-data/diabetes.csv
```

The `..` means:

```text
parent directory
```

So the notebook is looking for the CSV relative to the notebook's current working directory.

The final file is:

```text
diabetes.csv
```

inside:

```text
data/diabetes-data/
```

---

## Store the result in `df`

```python
df = pd.read_csv(...)
```

The loaded CSV data is stored in:

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
pandas DataFrame
   ↓
df
```

---

# 8. Display the First Rows

The notebook then uses:

```python
df.head()
```

## `head()`

`head()` is a pandas DataFrame method.

By default, it displays the first five rows.

Its purpose here is to quickly inspect the loaded dataset.

Conceptually:

```text
Full dataset
     ↓
   head()
     ↓
First 5 rows
```

---

# 9. Cell 3 — Split Data

The next Markdown heading is:

```markdown
## Split data
```

This section separates the dataset into:

```text
X = features
y = target
```

and then later divides them into training and test sets.

---

# 10. Cell 4 — Start Splitting the Data

The notebook begins with:

```python
print("Splitting data...")
```

This is another progress message.

It tells the user that the notebook is about to separate the input features and target.

---

# 11. Create `X` and `y`

The main line is:

```python
X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
```

This line creates two arrays:

```text
X
y
```

where:

```text
X = input features
y = target
```

---

# 12. Features Used by the Model

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

These become the model's input features.

The code is:

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

---

# 13. Meaning of Each Feature

The notebook uses the following feature names:

| Feature | Meaning |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `PlasmaGlucose` | Plasma glucose measurement |
| `DiastolicBloodPressure` | Diastolic blood-pressure measurement |
| `TricepsThickness` | Triceps skinfold-thickness measurement |
| `SerumInsulin` | Serum insulin measurement |
| `BMI` | Body Mass Index |
| `DiabetesPedigree` | Diabetes pedigree-related value |
| `Age` | Patient age |

These names are taken directly from the notebook's selected columns.

---

# 14. `.values`

The feature selection ends with:

```python
.values
```

For a pandas DataFrame, `.values` returns the underlying values as an array representation.

So:

```python
df[[...]].values
```

produces the feature data used for `X`.

---

# 15. Create the Target `y`

The second part is:

```python
df['Diabetic'].values
```

This selects the:

```text
Diabetic
```

column.

That column is the target variable.

The target is stored in:

```python
y
```

Therefore:

```text
X = input features
y = output/target
```

---

# 16. Complete Feature/Target Structure

The notebook effectively creates:

```text
                 Dataset
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
       X                        y
   Features                  Target
        │                       │
        ├─ Pregnancies          └─ Diabetic
        ├─ PlasmaGlucose
        ├─ DiastolicBloodPressure
        ├─ TricepsThickness
        ├─ SerumInsulin
        ├─ BMI
        ├─ DiabetesPedigree
        └─ Age
```

---

# 17. Cell 5 — Import `train_test_split`

The notebook uses:

```python
from sklearn.model_selection import train_test_split
```

This imports the `train_test_split` function from scikit-learn.

---

## `sklearn`

`sklearn` is the commonly used Python package name for **scikit-learn**.

Scikit-learn provides machine-learning algorithms and utilities.

---

## `model_selection`

This is the scikit-learn module containing tools for model selection and evaluation.

---

## `train_test_split`

This function divides data into training and testing subsets.

---

# 18. Train/Test Split

The notebook uses:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0
)
```

This is an important step.

---

# 19. Parameter: `X`

```python
X
```

contains all feature values.

It is supplied as the first positional argument.

---

# 20. Parameter: `y`

```python
y
```

contains all target labels.

It is supplied as the second positional argument.

---

# 21. Parameter: `test_size`

```python
test_size=0.30
```

This specifies that 30% of the data should be placed into the test set.

Conceptually:

```text
100% of data
      │
      ├───────────────┐
      ↓               ↓
   70% training     30% testing
```

Therefore:

```text
X_train → training features
X_test  → testing features
y_train → training labels
y_test  → testing labels
```

---

# 22. Parameter: `random_state`

```python
random_state=0
```

sets the random seed used by the splitting process.

A fixed random state makes the split reproducible.

If the same dataset and parameters are used again, the same split can be reproduced.

---

# 23. Four Outputs from `train_test_split()`

The function returns four objects:

```python
X_train
X_test
y_train
y_test
```

### `X_train`

Feature data used to train the model.

### `X_test`

Feature data used to evaluate the model.

### `y_train`

Correct labels corresponding to the training data.

### `y_test`

Correct labels corresponding to the test data.

Conceptually:

```text
                    X, y
                     │
                     ↓
             train_test_split()
                     │
            ┌────────┴────────┐
            ↓                 ↓
        Training            Testing
            │                 │
       X_train, y_train   X_test, y_test
```

---

# 24. Cell 6 — Train Model

The notebook introduces the training section with:

```markdown
## Train model
```

The model selected by the notebook is:

```text
Logistic Regression
```

---

# 25. Import Logistic Regression

The notebook uses:

```python
from sklearn.linear_model import LogisticRegression
```

This imports scikit-learn's `LogisticRegression` class.

---

# 26. What is Logistic Regression?

Despite its name, logistic regression is commonly used for **classification**.

For a binary classification problem, it estimates the probability of belonging to one of the classes.

Conceptually:

```text
Input features
      ↓
Logistic Regression
      ↓
Probability
      ↓
Class prediction
```

In this notebook, the prediction concerns:

```text
Diabetic
```

---

# 27. Print Training Message

The notebook uses:

```python
print("Training model...")
```

This displays:

```text
Training model...
```

It is simply a progress message.

---

# 28. Create and Train the Model

The notebook uses:

```python
model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)
```

This line both creates the model and trains it.

There are two main operations:

```python
LogisticRegression(...)
```

and:

```python
.fit(X_train, y_train)
```

---

# 29. Parameter: `C`

The notebook specifies:

```python
C=1/0.1
```

Python calculates:

```text
1 / 0.1 = 10
```

Therefore the actual value supplied to the `C` parameter is:

```text
C = 10
```

---

# 30. What Does `C` Mean?

In scikit-learn logistic regression, `C` is the inverse of regularization strength.

Conceptually:

```text
Higher C
   ↓
Weaker regularization

Lower C
   ↓
Stronger regularization
```

The notebook writes:

```python
C=1/0.1
```

which means:

```text
regularization-related value = 0.1
C = 10
```

---

# 31. Parameter: `solver`

The notebook uses:

```python
solver="liblinear"
```

The `solver` parameter selects the optimization algorithm used to train the logistic regression model.

The notebook specifically chooses:

```text
liblinear
```

---

# 32. `.fit(X_train, y_train)`

The final part is:

```python
.fit(X_train, y_train)
```

`fit()` trains the model using the supplied training data.

### `X_train`

Contains the training features.

### `y_train`

Contains the corresponding training labels.

After this operation:

```python
model
```

contains the fitted logistic regression model.

The complete process is:

```text
LogisticRegression(...)
        ↓
Create model
        ↓
.fit(X_train, y_train)
        ↓
Trained model
        ↓
model
```

---

# 33. Cell 8 — Evaluate Model

The notebook introduces the evaluation section with:

```markdown
## Evaluate model
```

The model is evaluated using:

```text
Accuracy
AUC
ROC Curve
```

---

# 34. Cell 9 — Import NumPy

The notebook uses:

```python
import numpy as np
```

This imports NumPy.

The alias:

```text
np
```

is used throughout the cell.

---

# 35. Generate Predictions

The notebook uses:

```python
y_hat = model.predict(X_test)
```

## `predict()`

`predict()` uses the trained model to generate class predictions for new input data.

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
Trained Logistic Regression
   ↓
y_hat
```

---

# 36. Calculate Accuracy

The notebook uses:

```python
acc = np.average(y_hat == y_test)
```

This line calculates accuracy manually.

It has two important parts:

```python
y_hat == y_test
```

and:

```python
np.average(...)
```

---

# 37. `y_hat == y_test`

This compares every predicted value with its actual value.

For example:

```text
Predicted: [1, 0, 1, 1]
Actual:    [1, 1, 1, 0]

Comparison:
           [True, False, True, False]
```

A `True` means the prediction is correct.

A `False` means the prediction is incorrect.

---

# 38. `np.average()`

The notebook then uses:

```python
np.average(...)
```

NumPy treats:

```text
True  ≈ 1
False ≈ 0
```

Therefore, averaging the Boolean comparison gives the proportion of correct predictions.

For example:

```text
[True, True, True, False]

Average = 3 / 4
        = 0.75
        = 75%
```

So:

```python
acc
```

contains the model's accuracy.

---

# 39. Print Accuracy

The notebook uses:

```python
print('Accuracy:', acc)
```

This displays the calculated accuracy.

The output will look conceptually like:

```text
Accuracy: 0.XX
```

The exact value depends on the data and model execution.

The notebook itself does not hard-code an accuracy result.

---

# 40. Cell 10 — Calculate AUC

The notebook imports:

```python
from sklearn.metrics import roc_auc_score
```

This imports the scikit-learn function:

```python
roc_auc_score
```

---

# 41. What is AUC?

AUC means:

```text
Area Under the ROC Curve
```

It summarizes how well a binary classifier separates the two classes over different thresholds.

A higher AUC generally indicates better class discrimination, but the notebook itself does not provide a specific interpretation or threshold for declaring a model acceptable.

---

# 42. Generate Probability Scores

The notebook uses:

```python
y_scores = model.predict_proba(X_test)
```

## `predict_proba()`

`predict_proba()` returns estimated probabilities for each class.

For example, a binary classifier might produce:

```text
Class 0    Class 1
  0.80       0.20
  0.25       0.75
  0.60       0.40
```

The result is stored in:

```python
y_scores
```

---

# 43. Select the Positive-Class Scores

The notebook uses:

```python
y_scores[:,1]
```

This is NumPy-style array indexing.

The two components are:

```text
:
```

and:

```text
1
```

### `:`

Selects all rows.

### `1`

Selects column index 1.

Therefore:

```python
y_scores[:,1]
```

selects the probability values for the second class for every test observation.

These scores are needed for ROC/AUC calculations.

---

# 44. Calculate ROC AUC

The notebook uses:

```python
auc = roc_auc_score(y_test,y_scores[:,1])
```

## `roc_auc_score()`

This function calculates the ROC AUC score.

It receives two important arguments:

### First argument

```python
y_test
```

The true labels.

### Second argument

```python
y_scores[:,1]
```

The model's scores/probabilities for the selected class.

The resulting AUC is stored in:

```python
auc
```

---

# 45. Print AUC

The notebook uses:

```python
print('AUC: ' + str(auc))
```

This combines text with the numerical AUC value.

---

## `str(auc)`

`str()` converts the AUC value into a string.

For example:

```python
str(0.82)
```

becomes:

```text
"0.82"
```

---

## String concatenation

The notebook uses:

```python
'AUC: ' + str(auc)
```

The `+` combines the two strings.

Conceptually:

```text
"AUC: "
   +
"0.82"
   ↓
"AUC: 0.82"
```

---

# 46. Cell 11 — Import ROC Curve Tools

The notebook imports:

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
```

There are two imports.

---

# 47. `roc_curve`

```python
from sklearn.metrics import roc_curve
```

imports the function used to calculate ROC curve points.

---

# 48. `matplotlib.pyplot`

```python
import matplotlib.pyplot as plt
```

imports Matplotlib's plotting interface.

The alias:

```text
plt
```

is used to create and configure the graph.

---

# 49. Calculate ROC Curve Values

The notebook uses:

```python
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
```

The `roc_curve()` function returns three arrays:

```text
fpr
tpr
thresholds
```

---

# 50. Parameter: `y_test`

```python
y_test
```

contains the actual labels from the test set.

These are the ground-truth values.

---

# 51. Parameter: `y_scores[:,1]`

```python
y_scores[:,1]
```

contains the model's probability/score for the selected class.

This allows the ROC curve to be evaluated at different classification thresholds.

---

# 52. Output: `fpr`

```python
fpr
```

means:

```text
False Positive Rate
```

It represents the false-positive rate at the different thresholds.

The formula is:

```text
FPR = False Positives / (False Positives + True Negatives)
```

---

# 53. Output: `tpr`

```python
tpr
```

means:

```text
True Positive Rate
```

It is also commonly called sensitivity or recall.

The formula is:

```text
TPR = True Positives / (True Positives + False Negatives)
```

---

# 54. Output: `thresholds`

```python
thresholds
```

contains the decision thresholds corresponding to the ROC points.

As the threshold changes, the corresponding FPR and TPR values change.

---

# 55. Create the Matplotlib Figure

The notebook uses:

```python
fig = plt.figure(figsize=(6, 4))
```

## `plt.figure()`

Creates a new Matplotlib figure.

The result is stored in:

```python
fig
```

---

## Parameter: `figsize`

```python
figsize=(6, 4)
```

specifies the figure dimensions.

The tuple represents:

```text
(width, height)
```

So the requested figure is:

```text
6 × 4 inches
```

---

# 56. Plot the Diagonal Reference Line

The notebook uses:

```python
plt.plot([0, 1], [0, 1], 'k--')
```

This draws the diagonal reference line.

---

## First argument

```python
[0, 1]
```

These are the x-axis coordinates.

---

## Second argument

```python
[0, 1]
```

These are the y-axis coordinates.

Together they create a line from:

```text
(0,0)
```

to:

```text
(1,1)
```

---

## Third argument: `'k--'`

```python
'k--'
```

is a Matplotlib line-format string.

It means:

```text
k  → black
-- → dashed line
```

Therefore the notebook draws a black dashed diagonal reference line.

The notebook's comment calls this the:

```text
50% line
```

---

# 57. Plot the Model ROC Curve

The notebook uses:

```python
plt.plot(fpr, tpr)
```

This plots the actual ROC curve generated by the model.

The arguments mean:

```text
x-axis → fpr
y-axis → tpr
```

Therefore:

```text
False Positive Rate
        ↓
       X-axis

True Positive Rate
        ↓
       Y-axis
```

---

# 58. Label the X-Axis

The notebook uses:

```python
plt.xlabel('False Positive Rate')
```

## `xlabel()`

Sets the x-axis label to:

```text
False Positive Rate
```

---

# 59. Label the Y-Axis

The notebook uses:

```python
plt.ylabel('True Positive Rate')
```

## `ylabel()`

Sets the y-axis label to:

```text
True Positive Rate
```

---

# 60. Set the Plot Title

The notebook uses:

```python
plt.title('ROC Curve')
```

## `title()`

Sets the chart title to:

```text
ROC Curve
```

---

# 61. Complete ROC Visualization

The final plotting section therefore does:

```text
Calculate ROC values
       ↓
fpr, tpr, thresholds
       ↓
Create figure
       ↓
Draw diagonal reference line
       ↓
Draw model ROC curve
       ↓
Label X axis
       ↓
Label Y axis
       ↓
Set title
```

The resulting visualization is a ROC curve.

---

# 62. Complete Notebook Workflow

The complete notebook can be understood as:

```text
                Diabetes CSV
                     │
                     ↓
               Read with pandas
                     │
                     ↓
                    df
                     │
                     ↓
          Separate features and target
                     │
            ┌────────┴────────┐
            ↓                 ↓
           X                  y
       Features             Diabetic
            │                 │
            └────────┬────────┘
                     ↓
            train_test_split()
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      Training                Testing
          │                     │
    X_train/y_train        X_test/y_test
          │                     │
          ↓                     │
 Logistic Regression             │
          │                     │
          └──────────┬──────────┘
                     ↓
                 Predictions
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       Accuracy                AUC
          │                     │
          │               predict_proba()
          │                     │
          │                     ↓
          │                ROC scores
          │                     │
          └──────────┬──────────┘
                     ↓
                 ROC Curve
```

---

# 63. Full Code Concept Map

```text
pandas
  │
  └── read_csv()
          ↓
         df
          │
          ├── Feature columns → X
          │
          └── Diabetic       → y
                    │
                    ↓
             train_test_split()
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
       Training             Testing
          │                   │
          ↓                   │
 LogisticRegression            │
          │                   │
          ↓                   ↓
        model             model.predict()
                              │
                              ↓
                            y_hat
                              │
                              ↓
                         Accuracy

model.predict_proba()
          │
          ↓
      y_scores
          │
          ├────────→ roc_auc_score()
          │              ↓
          │             AUC
          │
          └────────→ roc_curve()
                         │
                  ┌──────┼──────┐
                  ↓      ↓      ↓
                 fpr    tpr thresholds
                  │      │
                  └──┬───┘
                     ↓
               Matplotlib
                     ↓
                 ROC Curve
```

---

# 64. Line-by-Line Quick Reference

| Code | Purpose |
|---|---|
| `import pandas as pd` | Imports pandas as `pd` |
| `print("Reading data...")` | Displays a progress message |
| `pd.read_csv(...)` | Loads the diabetes CSV file |
| `df.head()` | Displays the first rows |
| `print("Splitting data...")` | Displays a progress message |
| `df[[...]]` | Selects the feature columns |
| `df['Diabetic']` | Selects the target column |
| `.values` | Gets the underlying array values |
| `from sklearn.model_selection import train_test_split` | Imports the train/test splitting function |
| `train_test_split(...)` | Divides data into training/testing subsets |
| `test_size=0.30` | Uses 30% of data for testing |
| `random_state=0` | Makes the split reproducible |
| `from sklearn.linear_model import LogisticRegression` | Imports logistic regression |
| `LogisticRegression(...)` | Creates the classification model |
| `C=1/0.1` | Sets `C` to 10 |
| `solver="liblinear"` | Selects the `liblinear` solver |
| `.fit(X_train, y_train)` | Trains the model |
| `import numpy as np` | Imports NumPy |
| `model.predict(X_test)` | Generates test predictions |
| `y_hat == y_test` | Compares predictions with actual labels |
| `np.average(...)` | Calculates the proportion of correct predictions |
| `print('Accuracy:', acc)` | Displays accuracy |
| `from sklearn.metrics import roc_auc_score` | Imports the AUC calculation function |
| `model.predict_proba(X_test)` | Generates class probabilities |
| `y_scores[:,1]` | Selects the second class probability |
| `roc_auc_score(...)` | Calculates ROC AUC |
| `str(auc)` | Converts AUC to a string |
| `from sklearn.metrics import roc_curve` | Imports ROC curve calculation |
| `import matplotlib.pyplot as plt` | Imports Matplotlib plotting tools |
| `roc_curve(...)` | Calculates FPR, TPR, and thresholds |
| `plt.figure(...)` | Creates a plot figure |
| `figsize=(6,4)` | Sets figure dimensions |
| `plt.plot([0,1],[0,1],'k--')` | Draws the diagonal reference line |
| `plt.plot(fpr,tpr)` | Draws the model ROC curve |
| `plt.xlabel(...)` | Labels the x-axis |
| `plt.ylabel(...)` | Labels the y-axis |
| `plt.title(...)` | Sets the chart title |

---

# 65. Important Viva / Exam Questions

## Q1. What is the purpose of this notebook?

To train a classification model that predicts diabetes using patient-related features and evaluate the model using accuracy, AUC, and an ROC curve.

---

## Q2. Is this supervised or unsupervised learning?

**Supervised learning**, because the dataset contains a known target:

```text
Diabetic
```

The model learns from features paired with known labels.

---

## Q3. Is this classification or regression?

**Classification.**

The model is `LogisticRegression`, used here to predict the diabetes class.

---

## Q4. What is the target column?

```text
Diabetic
```

It is extracted using:

```python
df['Diabetic'].values
```

---

## Q5. What are the input features?

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

## Q6. What percentage of the data is used for testing?

```text
30%
```

because:

```python
test_size=0.30
```

---

## Q7. Why is `random_state=0` used?

To make the train/test split reproducible.

---

## Q8. Which algorithm is used?

```text
Logistic Regression
```

using:

```python
LogisticRegression(...)
```

---

## Q9. What is the value of `C`?

The code is:

```python
C=1/0.1
```

Therefore:

```text
C = 10
```

---

## Q10. What does `C` control?

In scikit-learn logistic regression, `C` is the inverse of regularization strength.

```text
Higher C → weaker regularization
Lower C  → stronger regularization
```

---

## Q11. Which solver is used?

```text
liblinear
```

because:

```python
solver="liblinear"
```

---

## Q12. How is accuracy calculated?

The notebook uses:

```python
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
```

The prediction is compared with the actual label, and the average of the Boolean results gives the proportion of correct predictions.

---

## Q13. What is AUC?

AUC stands for:

```text
Area Under the ROC Curve
```

It summarizes the model's ability to distinguish between classes across different thresholds.

---

## Q14. Why is `predict_proba()` used?

Because ROC/AUC evaluation needs probability/score information rather than only the final class predictions.

The notebook uses:

```python
y_scores = model.predict_proba(X_test)
```

---

## Q15. What does `y_scores[:,1]` mean?

It selects:

```text
all rows
```

from:

```text
column index 1
```

of the probability array.

It therefore obtains the probability associated with the second class.

---

## Q16. What does `roc_curve()` return?

It returns:

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

## Q17. What is the x-axis of the ROC curve?

```text
False Positive Rate
```

---

## Q18. What is the y-axis of the ROC curve?

```text
True Positive Rate
```

---

## Q19. What is the purpose of the diagonal line?

The notebook draws:

```python
plt.plot([0, 1], [0, 1], 'k--')
```

This creates the diagonal 50% reference line shown in the notebook.

---

## Q20. Does this notebook use MLflow?

The notebook **mentions** a preference for using MLflow autologging in future work, but the supplied code does not import or use MLflow.

This is an important distinction.

---

# 66. Accuracy vs AUC

The notebook uses two different evaluation concepts.

## Accuracy

Accuracy answers:

> How many test predictions were classified correctly?

It is calculated as:

```python
np.average(y_hat == y_test)
```

Conceptually:

```text
Correct predictions
-------------------
All predictions
```

---

## AUC

AUC answers a different question:

> How well does the model distinguish between the two classes across different classification thresholds?

It uses the model's probability scores:

```python
y_scores[:,1]
```

and:

```python
roc_auc_score(y_test, y_scores[:,1])
```

Therefore:

```text
Accuracy → uses final class predictions
AUC      → uses class probability/score information
```

---

# 67. Accuracy vs ROC Curve

The notebook also creates an ROC curve.

```text
Accuracy
   ↓
One overall classification result
```

while:

```text
ROC Curve
   ↓
Performance across multiple thresholds
```

The ROC curve uses:

```text
FPR
vs.
TPR
```

---

# 68. Important Observation About the Notebook

The notebook introduction says that the data is already preprocessed and requires no feature engineering.

The supplied code also does not contain explicit feature-engineering or preprocessing operations.

Therefore, do not describe this notebook as performing:

```text
StandardScaler
OneHotEncoder
Missing-value imputation
Feature creation
Normalization
```

because none of those operations appear in the supplied notebook.

The notebook begins with already prepared data and proceeds directly to model training.

---

# 69. Important Observation About MLflow

The notebook's introduction says:

> Moving forward, there's a preference to use the autolog feature of MLflow to more easily deploy the model later on.

However, there is no code such as:

```python
import mlflow
```

or:

```python
mlflow.sklearn.autolog()
```

in the supplied notebook.

Therefore, MLflow is **mentioned as a future/preferred workflow**, not actually implemented in this notebook.

---

# 70. Final Mental Model

For an exam or viva, remember the complete workflow as:

```text
DIABETES CSV
     ↓
Pandas read_csv()
     ↓
DataFrame
     ↓
Select Features + Target
     ↓
X + y
     ↓
30% Test Split
     ↓
X_train / X_test
y_train / y_test
     ↓
Logistic Regression
     ↓
C = 10
Solver = liblinear
     ↓
Train with .fit()
     ↓
Predict with .predict()
     ↓
Accuracy
     ↓
predict_proba()
     ↓
AUC
     ↓
roc_curve()
     ↓
FPR + TPR + Thresholds
     ↓
Matplotlib
     ↓
ROC Curve
```

## One-Sentence Summary

> This lab loads an already-preprocessed diabetes dataset, separates patient features from the `Diabetic` target, trains a Logistic Regression classifier using a 70/30 train-test split, evaluates it using accuracy and ROC AUC, and visualizes its performance with an ROC curve.
