# Topic8FineTuning

This folder has two small fine-tuning examples:

- Exercise 1: a tiny LoRA demo that learns English to Pig Latin
- Exercise 2: a text-to-SQL workflow built around the `sql-create-context` dataset

The top-level files now keep the exercise number in the filename so it is easy to match them with the assignment steps.

## Setup

Install the packages you need for Tinker and the examples you want to run.

If you are using the text-to-SQL notebook, make sure the dataset JSON file is present in this folder.

## Exercise 1

Files:

- `ex1_pig_latin_lora_demo.py`
- `ex1_pig_latin_lora_demo.ipynb`

Purpose: a very small supervised fine-tuning example. It gives a few English phrases and Pig Latin targets, then trains a LoRA adapter for a few steps.

Example training pair:

```text
Input:  banana split
Output: anana-bay plit-say
```

Example prompt format:

```text
English: coffee break
Pig Latin:
```

Example model-style output after training:

```text
offee-cay eak-bray
```

## Exercise 2

Exercise 2 is the bigger text-to-SQL example.

### Main files

- `ex2_text_to_sql_finetuning_workflow.ipynb`
  Full notebook for dataset loading, split, prompt format, base evaluation, fine-tuning, and extra tests.

- `ex2_text_to_sql_dataset_preview.py`
  Small preview script that loads the dataset and shows counts plus one sample.

- `ex2_step1_prepare_train_test_split.py`
  Splits the dataset into training and test sets and saves the JSON files.

- `ex2_step2_prompt_format.txt`
  Simple prompt template for the text-to-SQL task.

- `ex2_step3_evaluate_base_model.py`
  Small script stub for base-model evaluation on the test split.

- `ex2_sql_execution_matcher.py`
  Compares generated SQL with expected SQL. It can do plain normalized string checks or execution-based checks with SQLite.

### Data files

- `ex2_sql_create_context_dataset.json`
- `ex2_train_data.json`
- `ex2_test_data.json`

### Example input and output

Dataset sample:

```text
Question: How many heads of departments are older than 56?
Context:  CREATE TABLE head (age INTEGER, name VARCHAR, ...)
Answer:   SELECT COUNT(*) FROM head WHERE age > 56
```

Prompt format:

```text
Table schema:
CREATE TABLE head (age INTEGER, name VARCHAR, ...)
Question: How many heads of departments are older than 56?
SQL: SELECT COUNT(*) FROM head WHERE age > 56
```

Example split output:

```text
Total examples: 78577
Training examples: 78377
Test examples: 200
Saved ex2_train_data.json and ex2_test_data.json
```

Example result summary from the notebook:

```text
Base model accuracy: 47.00% (94/200)
Fine-tuned model accuracy: 89.50% (179/200)
```

Example extra case:

```text
Question: What are the names of employees in the engineering department?
Context:  CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)
Fine-tuned SQL: SELECT id, name FROM employees WHERE department = 'Engineering'
Base SQL:      SELECT name FROM employees WHERE department = 'engineering'
```

## Notes

- `tinker-cookbook/` is a separate library folder and was left unchanged.
- Some scripts in this folder are small step files or partial workflow pieces, so the notebook is the clearest end-to-end reference.
