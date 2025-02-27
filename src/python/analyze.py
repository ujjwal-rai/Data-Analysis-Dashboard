import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def analyze_data(filepath):
    # Read the dataset
    df = pd.read_csv(filepath)
    
    # Store dataset info for chatbot context
    dataset_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'summary': df.describe().to_dict(),
        'sample': df.head(5).to_dict()
    }
    
    # Get datatypes summary
    datatypes = analyze_datatypes(df)
    
    # Get numerical summary
    numerical_summary = df.describe().to_dict()
    
    # Generate visualizations
    visualizations = generate_visualizations(df)
    
    # Generate correlation matrix
    correlation = generate_correlation_matrix(df)
    
    return {
        'datatypes': datatypes,
        'numerical_summary': numerical_summary,
        'visualizations': visualizations,
        'correlation': correlation,
        'dataset_info': dataset_info  # Add dataset info to response
    }

def analyze_datatypes(df):
    # Your existing analyze_datatypes function
    datatype_mapping = {
        'int64': 'Numerical',
        'float64': 'Numerical',
        'object': 'Categorical',
        'bool': 'Categorical',
        'datetime64[ns]': 'Datetime'
    }
    
    result = []
    for col in df.columns:
        result.append({
            'Column': col,
            'Pandas Dtype': str(df[col].dtype),
            'Custom Category': datatype_mapping.get(str(df[col].dtype), 'Other')
        })
    return result

def generate_visualizations(df):
    visualizations = []
    
    # Generate plots for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        
        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        visualizations.append({
            'title': f'Distribution of {col}',
            'data': [{
                'type': 'image',
                'source': f'data:image/png;base64,{image}'
            }]
        })
    
    return visualizations

def generate_correlation_matrix(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()
    
    return {
        'columns': numerical_cols.tolist(),
        'values': corr_matrix.values.tolist()
    }

if __name__ == "__main__":
    filepath = sys.argv[1]
    results = analyze_data(filepath)
    print(json.dumps(results)) 