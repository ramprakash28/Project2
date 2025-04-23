# Project 2

## Boosting Trees

Gradient Boosting Classifier from Scratch

## Project Overview

This project involves implementing a Gradient Boosting Classifier from scratch. The model utilizes decision trees as base learners, where each tree is trained to correct the errors made by previous trees using gradient descent. This model is designed for binary classification tasks.

## Team Members:
Chaitanya Datta Maddukuri – A20568393

Vamshi Krishna Cheeti – A20582646

Ram Prakash Bollam – A20561314

## Objective
The main objective is to implement the gradient boosting algorithm for classification from first principles. We aim to avoid using high-level libraries like scikit-learn for the core boosting functionality. Instead, we built the logic behind gradient boosting and decision tree construction ourselves.

Installation and Running Instructions
1. Clone the Repository

To clone the repository, use the following command:
git clone https://github.com/ramprakash28/Project2.git

  ## Virtual Environment Setup

  These steps assume you have Python 3.x (3.11 or 3.13) installed and are at the root of this project.

  **Create the venv**  
    python3 -m venv .venv
  # macOS
    source .venv/bin/activate
  # Optional
    pip install --upgrade pip setuptools wheel

2. Install Dependencies
Ensure Python is installed, then install all required dependencies using:

pip install -r requirements.txt

3. Generating Test Data
You can generate synthetic test data for regular and multicollinear cases by running:

For regular test data:

python generate_test_data.py \
  -N 100 \
  -m 2.5 3.0 \
  -b 5 \
  -scale 0.5 \
  -rnge -10 10 \
  -seed 42 \
  -output_file tests/test_data_regular.csv

For multicollinear data:

python generate_test_data.py \
  -N 100 \
  -m 0 0 0 0 0 0 \
  -b 0 \
  -scale 1 \
  -rnge -10 10 \
  -seed 42 \
  --multicollinear \
  -output_file tests/test_data_multicollinear.csv

4. Running the Test Suite

After installing dependencies, you can verify the implementation by running the test cases using:

pytest tests/test_GradientBoostingClassifier_regular.py

Collinear Data Handling:

We tested performance against multicollinear features using

pytest tests/test_GradientBoostingClassifier_multicollinear.py

## QnA :


1. What does the model you have implemented do and when should it be used?

The Gradient Boosting Classifier is explained as an ensemble method for classification. The answer covers the model's purpose, when it should be used (e.g., when you have complex relationships between features and target), and its strengths (e.g., handling non-linear decision boundaries and classification tasks).

2. How did you test your model to determine if it is working reasonably correctly?

The testing process is described clearly:

Regular Test Data: Used to validate basic functionality.

Multicollinear Data: To ensure the model can handle highly correlated features, a typical test scenario.

PyTest: The use of unit tests to ensure the individual components like tree-building and gradient calculation work as expected.

This question is answered thoroughly by explaining how you tested the model using different datasets and validating it using tests.

3. What parameters have you exposed to users of your implementation in order to tune performance? 

Parameters exposed:

N, m, b, scale, range, seed, multicollinear are all mentioned, with explanations about their role in controlling the model and dataset generation.

Usage Examples:

Example command for generating regular test data and multicollinear data are both provided, with the corresponding parameters shown.

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Inputs that the model struggles with:

The challenges with noisy data, outliers, and imbalanced datasets are explained.

Could you work around them?

Given more time, we could improve the model to handle these issues, e.g., by adding regularization techniques or better outlier detection.
