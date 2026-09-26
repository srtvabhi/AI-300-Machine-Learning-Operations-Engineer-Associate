# GitHub Operations Practice Exercise — ML Project in VS Code

> **Goal:** Practice the complete GitHub workflow using a small machine-learning-style Python project.
>
> **Scope:** Git + GitHub + GitHub Actions only. **No Azure ML or Azure services are used.**

---

# 🎯 Exercise Objective

By the end of this exercise, you will practice:

```text
Create ML project locally
        ↓
Initialize Git
        ↓
Create GitHub repository
        ↓
Push code
        ↓
Create feature branch
        ↓
Make changes
        ↓
Commit + Push
        ↓
Create Pull Request
        ↓
GitHub Actions runs tests
        ↓
Review checks
        ↓
Merge PR
        ↓
Pull latest main
        ↓
Create Git tag
        ↓
Create GitHub Release
```

---

# Part 1 — Create a Small ML Project

## Step 1: Create Project Folder

Open VS Code.

Create a folder:

```text
ml-github-demo
```

Open this folder in VS Code.

Initial structure:

```text
ml-github-demo/
```

---

## Step 2: Create `train.py`

Create:

```text
train.py
```

Add:

```python
def train_model():
    print("Training ML model...")
    accuracy = 0.92
    print(f"Model accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    train_model()
```

Run:

```bash
python train.py
```

Expected output:

```text
Training ML model...
Model accuracy: 92.00%
```

### What did we just do?

We created a simple ML-style training script:

```text
train.py
    ↓
Training
    ↓
Accuracy = 92%
```

---

# Part 2 — Add a Test

## Step 3: Create `test_train.py`

Create:

```text
test_train.py
```

Add:

```python
from train import train_model


def test_train_model():
    accuracy = train_model()
    assert accuracy >= 0.80
```

Install pytest if necessary:

```bash
pip install pytest
```

Run:

```bash
pytest
```

Expected:

```text
1 passed
```

### Why are we doing this?

Later, **GitHub Actions will automatically run this test** whenever we create or update a Pull Request.

---

# Part 3 — Add `.gitignore`

## Step 4: Create `.gitignore`

Create:

```text
.gitignore
```

Add:

```text
__pycache__/
*.pyc
.venv/
.env
.vscode/
```

### Why?

We don't want unnecessary or sensitive files going into GitHub.

For example:

```text
.env
```

could contain passwords, tokens, or other sensitive information.

---

# Part 4 — Initialize Git

## Step 5: Check Git

In the VS Code terminal:

```bash
git --version
```

You should see something similar to:

```text
git version 2.x.x
```

---

## Step 6: Initialize the repository

Run:

```bash
git init
```

You should see something similar to:

```text
Initialized empty Git repository
```

Now check:

```bash
git status
```

You should see files such as:

```text
train.py
test_train.py
.gitignore
```

---

# Part 5 — First Commit

## Step 7: Add Files

```bash
git add .
```

Check:

```bash
git status
```

You should see:

```text
Changes to be committed
```

---

## Step 8: Create Your First Commit

```bash
git commit -m "Initial ML project"
```
Explanation :
- Git takes the staged files and creates a commit.
- You can think of a commit as a snapshot of your project at a particular point in time.

Check the history:

```bash
git log --oneline
```

You should see something similar to:

```text
a82f31c Initial ML project
```

### Important Concept

```text
Working files
     ↓
git add
     ↓
Staging area
     ↓
git commit
     ↓
Git history
```

---

# Part 6 — Create GitHub Repository

## Step 9: Create a GitHub Repository

Go to GitHub and create a new repository.

Repository name:

```text
ml-github-demo
```

For this exercise:

- Don't add README
- Don't add `.gitignore`
- Don't add a license

We already created these locally.

After creating the repository, GitHub will provide a repository URL similar to:

```text
https://github.com/YOUR-USERNAME/ml-github-demo.git
```

---

# Part 7 — Connect Local Git to GitHub

## Step 10: Add GitHub Remote

Replace the URL with your repository URL:

```bash
git remote add origin https://github.com/YOUR-USERNAME/ml-github-demo.git
```

Check:

```bash
git remote -v
```

Expected:

```text
origin  https://github.com/YOUR-USERNAME/ml-github-demo.git
origin  https://github.com/YOUR-USERNAME/ml-github-demo.git
```

### What is `origin`?

`origin` is the conventional name for your remote GitHub repository.

Think:

```text
Local Git repository
        │
        │ origin
        ↓
GitHub repository
```

---

# Part 8 — Push `main`

## Step 11: Make Sure Your Branch Is `main`

Check:

```bash
git branch
```

If it shows `master`, rename it:

```bash
git branch -M main
```

Check again:

```bash
git branch
```

Expected:

```text
* main
```

---

## Step 12: Push to GitHub


```bash
git push -u origin main
```
Note  : Authenticate your self via browser or access token 

Step: Create a GitHub Personal Access Token (PAT)

A Personal Access Token (PAT) can be used to authenticate Git operations over HTTPS.

Steps

1. Sign in to GitHub using your `srtvabhi` account.

2. Go to:
   **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**

3. Click **Generate new token**.

4. Enter a token name, for example:
   ```text
   VS Code Git


Refresh your GitHub repository.

You should see:

```text
ml-github-demo

├── .gitignore
├── train.py
└── test_train.py
```

🎉 Your local ML project is now on GitHub.

---

# Part 9 — Create a Feature Branch

Now simulate a real developer workflow.

## Step 13: Create a Feature Branch

Suppose you want to improve the training code.

Run:

```bash
git switch -c feature/improve-training
```

Check:

```bash
git branch
```

Expected:

```text
* feature/improve-training
  main
```

### Important

You are now working on:

```text
feature/improve-training
```

instead of directly changing:

```text
main
```

---

# Part 10 — Make a Code Change

## Step 14: Modify `train.py`

Change:

```python
accuracy = 0.92
```

to:

```python
accuracy = 0.95
```

You can also change:

```python
print("Training ML model...")
```

to:

```python
print("Training improved ML model...")
```

Run:

```bash
python train.py
```

Expected:

```text
Training improved ML model...
Model accuracy: 95.00%
```

Run the test:

```bash
pytest
```

Expected:

```text
1 passed
```

---

# Part 11 — Commit the Feature

## Step 15: Check Changes

```bash
git status
```

Then inspect exactly what changed:

```bash
git diff
```

---

## Step 16: Commit

```bash
git add train.py
```

Then:

```bash
git commit -m "Improve model training"
```

Check:

```bash
git log --oneline
```

You should see something similar to:

```text
b72c4d1 Improve model training
a82f31c Initial ML project
```

---

# Part 12 — Push Feature Branch

## Step 17: Push Branch

```bash
git push -u origin feature/improve-training
```

GitHub should now show your new branch.

Conceptually:

```text
GitHub

main
 │
 └── feature/improve-training
```

---

# Part 13 — Create Pull Request

## Step 18: Open GitHub

Go to your repository.

Click:

**Compare & pull request**

Set:

```text
base: main

compare: feature/improve-training
```

PR title:

```text
Improve model training
```

Description:

```text
Updated the training logic and improved the reported accuracy.
```

Create the Pull Request.

---

# Part 14 — Add GitHub Actions

Now introduce automation.

Create this folder:

```text
.github/workflows/
```

Inside it create:

```text
tests.yml
```

Your project becomes:

```text
ml-github-demo/
│
├── train.py
├── test_train.py
├── .gitignore
│
└── .github/
    └── workflows/
        └── tests.yml
```

---

# Part 15 — Create GitHub Actions Workflow

## Step 19: Add Workflow

Put this into `tests.yml`:

```yaml
name: Python Tests

on:
  pull_request:
  push:
    branches:
      - main

jobs:
  test:

    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install pytest
        run: pip install pytest

      - name: Run tests
        run: pytest
```

---

# 🎓 Understand the Workflow

## `name`

```yaml
name: Python Tests
```

This is the workflow name.

---

## `on`

```yaml
on:
  pull_request:
```

This means:

> Start the workflow when a Pull Request event occurs.

We also have:

```yaml
push:
  branches:
    - main
```

Meaning:

> Run the workflow when code is pushed to `main`.

---

## `jobs`

```yaml
jobs:
  test:
```

We created a job called:

```text
test
```

---

## `runs-on`

```yaml
runs-on: ubuntu-latest
```

This specifies the runner.

Think:

```text
GitHub
   ↓
GitHub Actions
   ↓
Ubuntu virtual machine
```

---

## `steps`

```yaml
steps:
```

These are individual tasks:

```text
Step 1 → Checkout code
Step 2 → Setup Python
Step 3 → Install pytest
Step 4 → Run tests
```

---

# Part 16 — Commit the Workflow

## Step 20

Check:

```bash
git status
```

Add the workflow:

```bash
git add .github/workflows/tests.yml
```

Commit:

```bash
git commit -m "Add automated tests"
```

Push:

```bash
git push
```

Because you are already tracking the feature branch, `git push` is enough.

---

# Part 17 — Watch GitHub Actions

Go to your GitHub repository.

Click:

**Actions**

You should see:

```text
Python Tests
    ↓
Running
    ↓
Success
```

Click the workflow.

You will see:

```text
test
 ├── Checkout code
 ├── Set up Python
 ├── Install pytest
 └── Run tests
```

🎉 You have created your first GitHub Actions workflow.

---

# Part 18 — Understand the Complete PR Flow

Your current flow is:

```text
Developer
    ↓
Feature Branch
    ↓
Code Change
    ↓
Commit
    ↓
Push
    ↓
Pull Request
    ↓
GitHub Actions
    ↓
pytest
    ↓
PASS
```

This is the **quality gate** concept.

---

# Part 19 — Test a Failure

This is an important exercise.

Change your test:

```python
assert accuracy >= 0.99
```

Your model accuracy is:

```text
0.95
```

Therefore:

```text
0.95 >= 0.99
```

is false.

Commit:

```bash
git add test_train.py
git commit -m "Test stricter accuracy requirement"
git push
```

Go back to GitHub Actions.

You should see:

```text
Python Tests
     ↓
Failed
```

Click the failed job.

You will see the failed test.

### Why is this important?

The workflow detected a problem **before the code reaches `main`**.

This is the basic idea behind a CI quality gate.

---

# Part 20 — Fix the Test

Change:

```python
assert accuracy >= 0.99
```

back to:

```python
assert accuracy >= 0.80
```

Run locally:

```bash
pytest
```

Expected:

```text
1 passed
```

Commit:

```bash
git add test_train.py
git commit -m "Fix accuracy test"
git push
```

GitHub Actions automatically runs again.

Expected:

```text
Passed
```

---

# Part 21 — Merge the Pull Request

Go back to your Pull Request.

You should see:

```text
Checks passed
```

Click:

**Merge pull request**

Then:

**Confirm merge**

Your change is now in:

```text
main
```

---

# Part 22 — Update Your Local Main

Switch back:

```bash
git switch main
```

Pull the latest code:

```bash
git pull origin main
```

Your local `main` now contains the merged change.

---

# Part 23 — Delete the Feature Branch

After merging, the feature branch is no longer needed.

Delete the local branch:

```bash
git branch -d feature/improve-training
```

Delete the remote branch if it still exists:

```bash
git push origin --delete feature/improve-training
```

### Short-lived branch workflow

```text
Create branch
      ↓
Make change
      ↓
PR
      ↓
Merge
      ↓
Delete branch
```

---

# Part 24 — Create a Git Tag

Now practice Git tags.

Suppose this is your first stable version.

Create:

```bash
git tag v1.0.0
```

Check:

```bash
git tag
```

Expected:

```text
v1.0.0
```

Push the tag:

```bash
git push origin v1.0.0
```

Now GitHub has:

```text
v1.0.0
   ↓
Specific commit
   ↓
Specific version of your ML code
```

---

# Part 25 — Optional: Create a GitHub Release

Using GitHub CLI:

```bash
gh release create v1.0.0 \
  --title "Version 1.0.0" \
  --notes "First stable ML project version."
```

Check releases:

```bash
gh release list
```

Now you have practiced:

```text
Git tag
   ↓
GitHub Release
```

---

# 🧪 Final Project Structure

At the end, your repository should look approximately like:

```text
ml-github-demo/
│
├── train.py
├── test_train.py
├── .gitignore
│
└── .github/
    └── workflows/
        └── tests.yml
```

---

# 📚 What You Practiced

| Concept | What you did |
|---|---|
| Git repository | `git init` |
| GitHub repository | Created repo |
| Remote | `git remote add origin` |
| Branch | `git switch -c` |
| Commit | `git commit` |
| Push | `git push` |
| Pull | `git pull` |
| Pull Request | Created PR |
| GitHub Actions | Created workflow |
| Workflow trigger | `push`, `pull_request` |
| Job | `test` |
| Runner | `ubuntu-latest` |
| Steps | Checkout, Python, pytest |
| Unit testing | `pytest` |
| Quality gate | Workflow pass/fail |
| Branch protection | Can be configured on `main` |
| Merge | PR merged to `main` |
| Tag | `git tag v1.0.0` |
| GitHub Release | `gh release create` |

---

# 🎯 Most Important Commands

## Create repository locally

```bash
git init
```

## Check status

```bash
git status
```

## Create branch

```bash
git switch -c feature/my-change
```

## Add changes

```bash
git add .
```

## Commit

```bash
git commit -m "My change"
```

## Push

```bash
git push -u origin feature/my-change
```

## Create PR

```bash
gh pr create --base main
```

## Check PR

```bash
gh pr checks <PR-number>
```

## See Actions runs

```bash
gh run list
```

## Run workflow manually

```bash
gh workflow run train.yml
```

## Switch to main

```bash
git switch main
```

## Get latest code

```bash
git pull origin main
```

## Delete branch

```bash
git branch -d feature/my-change
```

## Create tag

```bash
git tag v1.0.0
```

## Push tag

```bash
git push origin v1.0.0
```

---

# ⭐ Final Mental Model

```text
LOCAL COMPUTER
      │
      │ git init
      ↓
    Git
      │
      │ git commit
      ↓
   Local History
      │
      │ git push
      ↓
    GITHUB
      │
      ├── Repository
      │
      ├── Branch
      │
      └── Pull Request
              │
              ↓
       GITHUB ACTIONS
              │
              ├── Runner
              │
              ├── Checkout
              │
              ├── Install
              │
              └── Test
                    │
              ┌─────┴─────┐
              ↓           ↓
            PASS         FAIL
              ↓           ↓
            Merge       Fix code
              ↓
            main
              ↓
           Tag v1.0.0
```

---

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

Workflow
   ↓
Automation instructions

Job
   ↓
Unit of work

Steps
   ↓
Individual tasks

pytest
   ↓
Tests the code

Quality Gate
   ↓
Pass / Fail before merge

main
   ↓
Stable branch

Tag
   ↓
Marks a version
```

# 🏆 Exercise Goal

If you can perform this entire sequence without blindly copying commands, you will have a practical foundation for the GitHub portion of an MLOps workflow:

> **Code → Git → Branch → Commit → Push → PR → GitHub Actions → Test → Merge → Tag**

**Azure ML is intentionally not involved anywhere in this exercise.**
