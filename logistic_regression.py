import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os
import ast
from groq import Groq
from pandasai import Agent

# Define your API key and client setup for Groq
client = Groq(api_key="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")

# Load CSV files and display basic info
dataset_path = sys.argv[1]
features_path = sys.argv[2]

dataset = pd.read_csv(dataset_path)
features_desc = pd.read_csv(features_path)

# Display info for verification
print("Dataset Info:")
print(dataset.info())
print("\nDataset Preview:")
print(dataset.head())

columns_of_dataset = dataset.columns.tolist()

# Request identifier column from Groq based on column names
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Which of the following columns can be an identifier {columns_of_dataset}, just write one column number for a zero-based indexing, nothing else",
        }
    ],
    model="llama3-8b-8192",
)
identifier = int(chat_completion.choices[0].message.content.strip())
print(f"Identifier column index (0-based): {identifier}")

# Ensure the identifier column is treated as a string (for student IDs, etc.)
dataset.iloc[:, identifier] = dataset.iloc[:, identifier].astype(str)

# Datatype analysis function
def analyze_datatypes(df):
    datatype_mapping = {
        'int64': 1, 'int32': 1, 'int16': 1, 'int8': 1,
        'float64': 1, 'float32': 1,
        'object': 2,  # Categorical
        'datetime64[ns]': 3,
        'bool': 2  # Boolean treated as categorical
    }
    column_types = {col: datatype_mapping.get(str(df[col].dtype), 4) for col in df.columns}
    new_df = pd.DataFrame([column_types])

    # Add padding columns if less than 10
    if len(new_df.columns) < 10:
        for i in range(10 - len(new_df.columns)):
            new_df[f'Extra_Col_{i}'] = 4

    return new_df

# Analyze datatypes and set identifier as "Identifier" type
datatype_df = analyze_datatypes(dataset)
datatype_df[columns_of_dataset[identifier]] = 0
print("\nDatatype Analysis:")
print(datatype_df)

# Request plot suggestions from Groq
instruction = (
    "Which types of plot (most important 5) should be drawn from such a data "
    "which will best summarise the dataset, just write each suggestion as a string, "
    "and write a list of strings, don't write anything else"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"I have a dataset with columns {columns_of_dataset}, {instruction}",
        }
    ],
    model="llama3-8b-8192",
)

suggestions = chat_completion.choices[0].message.content.strip()
print(f"Plot suggestions: {suggestions}")

# Convert plot suggestions to list format
suggestions = ast.literal_eval(suggestions)

# Initialize the PandasAI Agent for plotting
os.environ["PANDASAI_API_KEY"] = "$2a$10$evdTFIuQVdGYiNYJQUyHWeIVs7zP3Qu2.p1/3mRHJAsG692G3LV9S"
agent = Agent(dataset)

# Function to generate plots based on suggestions
def generate_plots_from_suggestions(suggestions):
    for suggestion in suggestions:
        # Send each suggestion to PandasAI agent for dynamic plotting
        response = agent.chat(f'Plot the following graph: {suggestion}')
        print(response)  # Display the response for each plot instruction

# Generate plots based on Groq-provided suggestions
generate_plots_from_suggestions(suggestions)
