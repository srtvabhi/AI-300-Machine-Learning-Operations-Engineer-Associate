# GitHub Actions + Azure ML — Easy Definitions with Commands

> Beginner-friendly MLOps glossary with Git, GitHub CLI, GitHub Actions, Azure ML, authentication, triggers, and practical examples.

## 1. GitHub

**Easy definition:**  
GitHub is a platform where we store and manage project code.

### Example

```text
GitHub
  ↓
my-ml-project
  ├── train.py
  ├── preprocess.py
  ├── requirements.txt
  └── job.yml
```

### Useful command

```bash
git clone https://github.com/<owner>/<repo>.git
```

**Remember:**  
> GitHub = Where we store and collaborate on code.

---

## 2. Git

**Easy definition:**  
Git is a version-control system that tracks changes in your code.

### Example

```text
Commit 1 → train.py created
Commit 2 → training algorithm changed
Commit 3 → bug fixed
```

### Useful commands

```bash
git status
git add .
git commit -m "Fix training logic"
git log
```

**Remember:**  
> Git = Tracks code changes.

---

## 3. Repository

**Easy definition:**  
A repository, or **repo**, is your project stored in GitHub.

### Example

```text
GitHub
   ↓
my-ml-project
   ├── train.py
   ├── test.py
   ├── job.yml
   └── requirements.txt
```

### Useful commands

```bash
git clone https://github.com/<owner>/<repo>.git
git remote -v
```

**Remember:**  
> Repository = Home for your project.

---

## 4. Branch

**Easy definition:**  
A branch is a separate line of development.

### Example

```text
main
  │
  └── feature/add-bmi
```

### Commands

```bash
git checkout -b feature/add-bmi
git switch -c feature/add-bmi
git branch
```

**Remember:**  
> Branch = Separate workspace for a change.

---

## 5. Main Branch

**Easy definition:**  
`main` is normally the stable branch of the project.

### Commands

```bash
git switch main
git pull origin main
```

---

## 6. Feature Branch

**Easy definition:**  
A short-lived branch used to develop one specific feature or fix.

### Commands

```bash
git switch -c feature/improve-training
git push -u origin feature/improve-training
```

---

## 7. Pull Request — PR

**Easy definition:**  
A Pull Request asks the team to review and merge your branch into another branch, usually `main`.

### GitHub CLI

```bash
gh pr create --base main --head feature/improve-training
gh pr list
gh pr view <PR-number>
```

**Remember:**  
> PR = "Please review and merge my changes."

---

## 8. Trunk-Based Development

**Easy definition:**  
A development approach where `main` stays stable and developers use short-lived branches.

### Typical commands

```bash
git switch main
git pull
git switch -c feature/new-model

git add .
git commit -m "Add new model"
git push -u origin feature/new-model

gh pr create --base main
```

---

## 9. GitHub Actions

**Easy definition:**  
GitHub Actions is GitHub's automation system.

### Example

```text
Pull Request
     ↓
GitHub Actions
     ↓
Tests
     ↓
Linting
     ↓
Result
```

Workflow files normally live here:

```text
.github/workflows/
```

### Useful commands

```bash
gh run list
gh run view <run-id>
```

**Remember:**  
> GitHub = Stores code  
> GitHub Actions = Automates work

---

## 10. Workflow

**Easy definition:**  
A workflow is an automated process defined in a YAML file.

### Example

```text
.github/workflows/train.yml
```

```yaml
name: Train Model

on:
  push:
    branches:
      - main

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
```

**Remember:**  
> Workflow = Automation recipe.

---

## 11. YAML

**Easy definition:**  
YAML is a human-readable configuration format.

### Example

```yaml
name: Test

on:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
```

Here:

```text
name → Workflow name
on   → Trigger
jobs → Work to perform
```

---

## 12. Trigger

**Easy definition:**  
A trigger is the event that starts a GitHub Actions workflow.

Common triggers:

```text
push
pull_request
workflow_dispatch
schedule
repository_dispatch
```

---

## 13. `push`

**Easy definition:**  
Runs a workflow when code is pushed to a branch.

### Command

```bash
git push origin main
```

### Workflow

```yaml
on:
  push:
    branches:
      - main
```

**Typical use:**  
Train or register a model after an approved change reaches `main`.

---

## 14. `pull_request`

**Easy definition:**  
Runs a workflow when a Pull Request is created or updated.

### Workflow

```yaml
on:
  pull_request:
```

### Command

```bash
gh pr checks <PR-number>
```

**Typical use:**  
Validate code before merging.

---

## 15. `workflow_dispatch`

**Easy definition:**  
Allows a person to manually start a workflow from GitHub.

### Workflow

```yaml
on:
  workflow_dispatch:
```

### GitHub CLI

```bash
gh workflow run train.yml
gh workflow run train.yml --ref main
```

**Typical use:**  
Manually start model training.

---

## 16. `schedule`

**Easy definition:**  
Runs a workflow automatically according to a schedule.

### Example

```yaml
on:
  schedule:
    - cron: "0 2 * * 0"
```

**Typical use:**  
Regular retraining.

---

## 17. `repository_dispatch`

**Easy definition:**  
Allows an external application to tell GitHub to start a workflow.

### GitHub CLI

```bash
gh api \
  --method POST \
  /repos/<owner>/<repo>/dispatches \
  -f event_type=retrain
```

### Workflow

```yaml
on:
  repository_dispatch:
    types:
      - retrain
```

**Typical use:**  
An Azure-side event starts ML retraining.

---

## 18. Job

**Easy definition:**  
A job is a unit of work inside a GitHub Actions workflow.

### YAML

```yaml
jobs:
  validate:
    runs-on: ubuntu-latest

  train:
    runs-on: ubuntu-latest
```

---

## 19. Runner

**Easy definition:**  
A runner is the computer or virtual machine where a GitHub Actions job executes.

### Example

```yaml
runs-on: ubuntu-latest
```

**Remember:**  
> Runner = Computer doing the work.

---

## 20. Step

**Easy definition:**  
A step is an individual task inside a job.

### Example

```yaml
steps:
  - uses: actions/checkout@v4

  - name: Install dependencies
    run: pip install -r requirements.txt

  - name: Run tests
    run: pytest
```

---

## 21. Linting

**Easy definition:**  
Linting checks code quality and coding style.

### Command

```bash
flake8 .
```

### GitHub Actions

```yaml
- name: Run linting
  run: flake8 .
```

**Remember:**  
> Linting = Check code quality and style.

---

## 22. Unit Test

**Easy definition:**  
A unit test checks whether a small piece of code behaves correctly.

### Python example

```python
def add(a, b):
    return a + b
```

```python
assert add(2, 3) == 5
```

### Run tests

```bash
pytest
```

### GitHub Actions

```yaml
- name: Run tests
  run: pytest
```

**Remember:**  
> Unit test = Check code behavior.

---

## 23. Quality Gate

**Easy definition:**  
A quality gate is a required check that must pass before code can merge.

```text
Pull Request
     ↓
GitHub Actions
     ↓
Lint + Tests
     ↓
PASS → Merge
FAIL → Block merge
```

### Check PR status

```bash
gh pr checks <PR-number>
```

**Remember:**  
> Quality gate = Required pass/fail checkpoint.

---

## 24. Branch Protection

**Easy definition:**  
Branch protection is a GitHub rule that controls what must happen before changes can enter a protected branch such as `main`.

### Example API command

```bash
gh api repos/<owner>/<repo>/branches/main/protection
```

**Important distinction:**

> **GitHub Actions performs the check.**  
> **Branch protection makes the check mandatory.**

---

## 25. Azure

**Easy definition:**  
Azure is Microsoft's cloud platform.

It provides services for:

```text
Compute
Storage
Databases
AI/ML
Networking
```

---

## 26. Azure Machine Learning — Azure ML

**Easy definition:**  
Azure Machine Learning is Microsoft's cloud platform for building, training, managing, and deploying ML models.

```text
GitHub Actions
      ↓
Azure ML
      ↓
Training
      ↓
Model
```

**Remember:**  
> Azure ML = Where the ML workload runs.

---

## 27. Azure ML Job

**Easy definition:**  
An Azure ML job is a workload that Azure Machine Learning executes.

It can define:

```text
Code
Environment
Compute
Inputs
Parameters
Outputs
```

---

## 28. Command Job

**Easy definition:**  
A command job runs a command or script on Azure ML compute.

```text
train.py
   ↓
Command Job
   ↓
Azure Compute
   ↓
Training
```

### Azure CLI

```bash
az ml job create --file job.yml
```

**Remember:**  
> Command job = One main ML workload/script.

---

## 29. Pipeline Job

**Easy definition:**  
A pipeline job connects multiple ML components.

```text
Prepare Data
     ↓
Train
     ↓
Evaluate
     ↓
Register Model
```

### Azure CLI

```bash
az ml job create --file pipeline.yml
```

**Remember:**

> Command job = Single workload  
> Pipeline job = Multiple connected workloads

---

## 30. Component

**Easy definition:**  
A component is a reusable ML step.

```text
Prepare Data Component
        ↓
Training Component
        ↓
Evaluation Component
```

Components can be reused in different pipelines.

---

## 31. Azure Compute

**Easy definition:**  
Azure Compute is the cloud resource that actually performs the ML computation.

```text
Azure ML Job
     ↓
Azure Compute
     ↓
Python code executes
```

> **Azure ML manages the job; Compute performs the computation.**

---

## 32. Job YAML

**Easy definition:**  
A YAML file that describes an Azure ML job.

```text
job.yml
   ↓
Code
Compute
Environment
Parameters
   ↓
Azure ML
```

### Submit the job

```bash
az ml job create --file job.yml
```

---

## 33. Azure CLI

**Easy definition:**  
Azure CLI is a command-line tool used to interact with Azure.

### Example

```text
GitHub Actions
      ↓
Azure CLI
      ↓
Azure ML
```

### Command

```bash
az ml job create --file job.yml
```

**Remember:**  
> Azure CLI = Command-line way to control Azure.

---

## 34. Authentication

**Easy definition:**  
Authentication answers:

> **"Who are you?"**

```text
GitHub Actions
      ↓
Microsoft Entra ID
      ↓
Identity verified
      ↓
Azure ML
```

---

## 35. Authorization

**Easy definition:**  
Authorization answers:

> **"What are you allowed to do?"**

Example:

```text
Submit Azure ML job → Allowed
Delete entire subscription → Not allowed
```

**Simple distinction:**

> Authentication = **Who are you?**  
> Authorization = **What can you do?**

---

## 36. Microsoft Entra ID

**Easy definition:**  
Microsoft Entra ID is Microsoft's identity and access-management service.

```text
GitHub Actions
      ↓
Microsoft Entra ID
      ↓
Verify identity
      ↓
Azure ML access
```

---

## 37. Service Principal

**Easy definition:**  
A service principal is a **non-human identity** used by an application or automation system.

```text
GitHub Actions
      ↓
Service Principal
      ↓
Azure
```

The service principal can be assigned an Azure role.

---

## 38. Least Privilege

**Easy definition:**  
Give an identity **only the permissions it actually needs**.

### Too broad

```text
GitHub Actions
      ↓
Entire Azure subscription
```

### Better

```text
GitHub Actions
      ↓
Service Principal
      ↓
Required Azure ML workspace
```

**Remember:**

> Minimum required access = **Least privilege**

---

## 39. GitHub Secrets

**Easy definition:**  
GitHub Secrets store sensitive information that should not be exposed in source code.

### Examples

```text
Client secret
API token
Credential
Password
```

### GitHub CLI

```bash
gh secret set AZURE_CLIENT_SECRET
gh secret set AZURE_CLIENT_SECRET --env production
gh secret list
```

**Remember:**

> Sensitive value → **Secret**

---

## 40. GitHub Variables

**Easy definition:**  
GitHub Variables store non-sensitive configuration.

### Examples

```text
AZURE_ML_WORKSPACE
RESOURCE_GROUP
COMPUTE_NAME
```

### GitHub CLI

```bash
gh variable set AZURE_ML_WORKSPACE --body "ml-workspace"
gh variable list
```

**Remember:**

> Non-sensitive configuration → **Variable**

---

## 41. OIDC

**Full name:** OpenID Connect

**Easy definition:**  
OIDC allows GitHub Actions to authenticate to Azure using a **short-lived token** instead of storing a long-lived client secret.

### Traditional approach

```text
Long-lived secret
      ↓
Stored in GitHub
      ↓
Potential exposure
```

### OIDC approach

```text
GitHub Actions
      ↓
Short-lived OIDC token
      ↓
Microsoft Entra ID
      ↓
Azure access token
      ↓
Azure ML
```

### GitHub Actions permission

```yaml
permissions:
  id-token: write
```

**Remember:**  
> OIDC = Short-lived identity instead of a long-lived secret.

---

## 42. Workload Identity Federation

**Easy definition:**  
Workload identity federation establishes trust between GitHub and Microsoft Entra ID so GitHub Actions can access Azure without storing a long-lived client secret.

```text
GitHub
   ↕
Trust relationship
   ↕
Microsoft Entra ID
```

Then:

```text
GitHub Actions
      ↓
Short-lived token
      ↓
Entra ID
      ↓
Azure access
```

---

## 43. Git Tracking

**Easy definition:**  
Git tracking answers:

> **"Which version of my code produced this ML training job?"**

```text
Repository
   ↓
Branch
   ↓
Commit
   ↓
Training Job
```

### Useful commands

```bash
git branch --show-current
git rev-parse HEAD
git log -1
```

Example:

```text
Repository: ml-project
Branch: main
Commit: abc123
Training Job: job_456
```

This gives you **traceability**.

---

## 44. Traceability

**Easy definition:**  
Traceability means being able to trace an ML result back to the exact source code that produced it.

```text
Model v10
   ↓
Training Job #456
   ↓
Commit abc123
   ↓
train.py version
```

Question answered:

> **"Which code produced this model?"**

---

## 45. Authentication ≠ Git Tracking

This is one of the **most important distinctions**.

### Git tracking

```text
Repository
   ↓
Branch
   ↓
Commit
   ↓
Training Job
```

Answers:

> **Which code produced this job?**

### Authentication

```text
GitHub Actions
      ↓
Microsoft Entra ID
      ↓
Azure ML
```

Answers:

> **Does this workflow have permission to submit the job?**

Therefore:

```text
Git tracking
     ↓
TRACEABILITY
```

and:

```text
Authentication
     ↓
PERMISSION
```

They solve different problems.

---

## 46. Event

**Easy definition:**  
An event is something that happened and can start an automated process.

```text
New data arrives
      ↓
Event
      ↓
Start retraining
```

---

## 47. Azure Event Grid

**Easy definition:**  
Azure Event Grid detects and delivers events from Azure services.

```text
New file arrives
      ↓
Azure Storage
      ↓
Event Grid
      ↓
"New data arrived!"
```

---

## 48. Azure Function

**Easy definition:**  
Azure Functions lets you run small pieces of code in Azure without managing a traditional server.

```text
Event Grid
    ↓
Azure Function
    ↓
GitHub REST API
```

The Function acts as an **intermediary**.

---

## 49. Azure Logic Apps

**Easy definition:**  
Azure Logic Apps helps connect different services and automate workflows.

```text
Event Grid
    ↓
Logic App
    ↓
GitHub API
    ↓
repository_dispatch
```

---

## 50. GitHub REST API

**Easy definition:**  
The GitHub REST API allows another application to interact with GitHub programmatically.

```text
Azure Function
      ↓
GitHub REST API
      ↓
repository_dispatch
      ↓
GitHub Actions
```

### GitHub CLI

```bash
gh api \
  --method POST \
  /repos/<owner>/<repo>/dispatches \
  -f event_type=retrain
```

---

## 51. Repository Dispatch

**Easy definition:**  
`repository_dispatch` is a GitHub event that an external system can use to start a workflow.

### Workflow

```yaml
on:
  repository_dispatch:
    types:
      - retrain
```

### External trigger

```text
Azure
  ↓
GitHub API
  ↓
repository_dispatch
  ↓
GitHub Actions
```

---

## 52. Train

**Easy definition:**  
Training means using data and an algorithm to create a trained ML model.

```text
Training Data
      +
Algorithm
      ↓
Trained Model
```

---

## 53. Evaluate

**Easy definition:**  
Evaluation measures how well the trained model performs.

```text
Trained Model
      ↓
Test Data
      ↓
Accuracy = 92%
```

---

## 54. Register Model

**Easy definition:**  
Registering a model means storing and versioning the trained model so it can be managed and used later.

```text
Training
   ↓
Evaluation
   ↓
Model v1
   ↓
Register
```

---

## 55. Model Registry

**Easy definition:**  
A model registry is a managed place to store and version ML models.

> Model Registry = Version control for trained models

### Example

```text
Model
 ├── v1
 ├── v2
 └── v3
```

---

## 56. MLOps

**Easy definition:**  
MLOps means applying software-engineering, automation, and operational practices to the ML lifecycle.

### Simplified lifecycle

```text
PLAN
  ↓
CODE
  ↓
VALIDATE
  ↓
TRAIN
  ↓
EVALUATE
  ↓
REGISTER
  ↓
DEPLOY
  ↓
MONITOR
  ↓
RETRAIN
```

This module mainly connects:

```text
CODE
  ↓
VALIDATE
  ↓
TRAIN
```

---

# 57. Complete Picture

```text
Developer
    ↓
Feature Branch
    ↓
Pull Request
    ↓
GitHub Actions
    │
    ├── Linting
    ├── Unit Tests
    │
    └── Authentication
             ↓
       Microsoft Entra ID
             ↓
       Azure Machine Learning
             ↓
        Submit Job
             ↓
       Train / Evaluate
             ↓
       Register Model
```

The source version can also be tracked:

```text
Repository
    ↓
Branch
    ↓
Commit
    ↓
Training Job
    ↓
Model
```

---

# 🎯 The 5 Questions to Remember

## 1. Where is my code?

### GitHub

```text
GitHub Repository
```

Useful commands:

```bash
git clone <repo-url>
git status
```

---

## 2. What changed?

### Git

```text
Branch
   ↓
Commit
```

Useful commands:

```bash
git switch -c feature/my-change
git add .
git commit -m "Update training code"
git push -u origin feature/my-change
```

---

## 3. When should automation start?

### GitHub Actions triggers

```text
push
pull_request
workflow_dispatch
schedule
repository_dispatch
```

Example:

```bash
gh workflow run train.yml
```

External event:

```bash
gh api \
  --method POST \
  /repos/<owner>/<repo>/dispatches \
  -f event_type=retrain
```

---

## 4. Can GitHub Actions access Azure?

### Authentication + Authorization

```text
GitHub Actions
      ↓
OIDC / Service Principal
      ↓
Microsoft Entra ID
      ↓
Azure ML
```

Modern approach:

```text
GitHub Actions
      ↓
Short-lived OIDC token
      ↓
Microsoft Entra ID
      ↓
Azure access token
      ↓
Azure ML
```

---

## 5. What should Azure ML run?

### Azure ML Job

```text
Command Job
     OR
Pipeline Job
```

Submit with Azure CLI:

```bash
az ml job create --file job.yml
```

---

# ⭐ One-Line Trainer Summary

> **GitHub stores the ML code, Git tracks its versions, branches isolate changes, Pull Requests enable review, GitHub Actions automates the process, triggers decide when it runs, runners execute the steps, Microsoft Entra ID authenticates the workflow, OIDC provides short-lived identity, Azure ML runs the job, and Git tracking tells us exactly which code produced the model.**

---

# 🧠 One Final Mental Model

```text
                 GITHUB
                   │
        ┌──────────┴──────────┐
        │                     │
    Repository             Actions
        │                     │
   Branch / Commit       Trigger
        │                     │
        │              Workflow / Job
        │                     │
        │                  Runner
        │                     │
        │             Tests / Linting
        │                     │
        │                Authentication
        │                     │
        └──────────────┬──────┘
                       ↓
                Microsoft Entra ID
                       ↓
                 Azure Machine
                   Learning
                       ↓
              Command / Pipeline Job
                       ↓
                Train / Evaluate
                       ↓
                 Register Model
                       ↓
                  Model Registry
```

# 🚀 Simplest Version for Beginners

```text
GitHub
   ↓
Stores code

Git
   ↓
Tracks changes

Branch
   ↓
Work safely on a change

Pull Request
   ↓
Ask for review

GitHub Actions
   ↓
Automates the work

Trigger
   ↓
Decides WHEN to run

Runner
   ↓
Computer that executes the workflow

Microsoft Entra ID
   ↓
Authenticates the workflow

OIDC / Service Principal
   ↓
Provides Azure identity

Azure ML
   ↓
Runs the ML workload

Command Job
   ↓
One main workload

Pipeline Job
   ↓
Multiple connected workloads

Git Tracking
   ↓
Provides traceability

Model Registry
   ↓
Stores model versions
```

## Final Mental Shortcut

> **GitHub stores → Git tracks → Branch isolates → PR reviews → Actions automates → Trigger starts → Runner executes → Entra authenticates → Azure ML runs → Registry stores → Git tracking provides traceability.**
