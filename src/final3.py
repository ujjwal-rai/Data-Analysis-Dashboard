# Page implementation

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
from groq import Groq
import ast
from PIL import Image

# Define a mapping for datatype categories
DATATYPE_MAPPING = {
    'int64': 'Numerical',
    'int32': 'Numerical',
    'float64': 'Numerical',
    'float32': 'Numerical',
    'object': 'Categorical',
    'bool': 'Categorical',
    'datetime64[ns]': 'Datetime',
    'category': 'Categorical'
}

DEFAULT_CATEGORY = 'Other'  # Fallback for unknown datatypes

# Initialize Groq client
client = Groq(
    api_key="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",
)

def get_identifier_column(columns):
    """
    Uses an AI model to determine the identifier column based on column names.
    Args:
        columns: List of column names in the dataset.
    Returns:
        The index (integer) of the identifier column.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Which of the following columns can be an identifier {columns}, just write one column number for a zero based indexing, nothing else",
            }
        ],
        model="llama3-8b-8192",
    )
    identifier_index = int(chat_completion.choices[0].message.content)
    return identifier_index

def analyze_datatypes(df, identifier_index=None):
    """
    Analyzes datatypes of DataFrame columns and maps them to custom categories.
    Args:
        df: Input Pandas DataFrame.
        identifier_index: Index of the identifier column (if any).
    Returns:
        A DataFrame summarizing the datatypes of the columns.
    """
    # Map datatypes using the predefined mapping
    datatype_summary = df.dtypes.astype(str).map(DATATYPE_MAPPING).fillna(DEFAULT_CATEGORY)
    
    # If identifier index is provided, update its category
    if identifier_index is not None:
        identifier_column = df.columns[identifier_index]
        datatype_summary[identifier_column] = 'Identifier'
    
    # Create a DataFrame to display results
    result_df = pd.DataFrame({
        'Column': df.columns,
        'Pandas Dtype': df.dtypes.astype(str),
        'Custom Category': datatype_summary
    }).reset_index(drop=True)

    return result_df

def handle_missing_values(df, datatype_df):
    """
    Handles missing values based on column data types.
    Args:
        df: Input DataFrame.
        datatype_df: DataFrame summarizing the datatype categories.
    Returns:
        DataFrame with missing values handled.
    """
    for col, dtype in zip(datatype_df['Column'], datatype_df['Custom Category']):
        if dtype == 'Numerical':
            # Impute numerical columns with the median value
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        elif dtype == 'Categorical':
            # Impute categorical columns with the mode
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        else:
            # For other types, we can drop rows with missing values
            df.dropna(subset=[col], inplace=True)
    print(df.head())
    return df

def remove_duplicates(df, datatype_df):
    """
    Removes duplicate rows based on column data types.
    Args:
        df: Input DataFrame.
        datatype_df: DataFrame summarizing the datatype categories.
    Returns:
        DataFrame with duplicates removed.
    """
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Optionally, remove duplicate values in categorical columns
    for col, dtype in zip(datatype_df['Column'], datatype_df['Custom Category']):
        if dtype == 'Categorical':
            # Check for duplicate categories and decide on handling
            df[col] = df[col].str.strip()  # Stripping whitespace in categorical values
    
    return df

def data_cleaning(df, datatype_df):
    """
    Clean the DataFrame by handling missing values, removing duplicates, etc.
    Args:
        df: Input DataFrame.
        datatype_df: DataFrame summarizing the datatype categories.
    Returns:
        Cleaned DataFrame.
    """
    # Handle missing values
    df = handle_missing_values(df, datatype_df)

    # Remove duplicates
    df = remove_duplicates(df, datatype_df)

    # Any additional cleaning steps can be added here (e.g., outlier handling, normalization, etc.)
    print(df.head())
    return df   

def summarize_and_visualize(df, datatype_df):
    """
    Generates five-number summary for numerical features and plots histograms for categorical features.
    Args:
        df: Input cleaned DataFrame.
        datatype_df: DataFrame summarizing the datatype categories.
    """
    # Get numerical and categorical columns
    numerical_columns = datatype_df[datatype_df['Custom Category'] == 'Numerical']['Column'].tolist()
    categorical_columns = datatype_df[datatype_df['Custom Category'] == 'Categorical']['Column'].tolist()

    # Five-number summary for numerical features
    if numerical_columns:
        st.write("### Five-Number Summary for Numerical Features")
        st.write(df[numerical_columns].describe().T)  # Transpose for better readability

        # Optional: Visualizations for numerical features
        for col in numerical_columns:
            st.write(f"*Distribution for {col}:*")
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, color='skyblue', bins=30)
            plt.title(f"Histogram for {col}")
            st.pyplot(plt)
            plt.clf()  # Clear figure after displaying each plot

    # Plotting histograms for categorical features
    if categorical_columns:
        st.write("### Frequency Distribution for Categorical Features")
        for col in categorical_columns:
            # Get the frequency of categories and select top 7 if needed
            category_counts = df[col].value_counts()
            top_categories = category_counts.head(7)  # Top 7 categories

            # Display top 7 categories
            st.write(f"*Top Categories for {col}:*")
            st.write(top_categories)

            # Plot histogram for the top categories
            plt.figure(figsize=(8, 4))
            sns.countplot(x=df[col], order=top_categories.index, palette='Set2')
            plt.title(f"Histogram for Top Categories of {col}")
            st.pyplot(plt)
            plt.clf()  # Clear figure after displaying each plot

def draw_correlation_heatmap(df, datatype_df):
    """
    Draws a heatmap of correlation for all numerical features in the dataset.
    
    Args:
        df: Input cleaned DataFrame.
        datatype_df: DataFrame summarizing the datatype categories.
    """
    # Get numerical columns
    numerical_columns = datatype_df[datatype_df['Custom Category'] == 'Numerical']['Column'].tolist()
    
    if numerical_columns:
        # Calculate correlation matrix
        correlation_matrix = df[numerical_columns].corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
        plt.title("Correlation Heatmap for Numerical Features")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)
    else:
        st.write("No numerical features available for correlation heatmap.")

def generate_report_with_pandasai(df, agent, groq_client):
    """
    Generate a report using PandasAI based on AI-suggested plots.
    
    Args:
        df: The uploaded DataFrame.
        agent: Initialized PandasAI Agent.
        groq_client: Initialized Groq client for generating plot suggestions.
    """
    columns_of_dataset = df.columns.tolist()
    instruction3 = 'Suggest 2 most important graphs that should be plotted from this dataset. Just write each suggestion as a string, and write a list of strings, dont write anything else'
    
    # Generate suggestions for plotting
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"I have a dataset with columns {columns_of_dataset}, {instruction3}",
            }
        ],
        model="llama3-8b-8192",
    )
    
    # Extract suggestions from Groq response
    suggestions = chat_completion.choices[0].message.content
    suggestions = ast.literal_eval(suggestions)  # Convert string to list
    
    for suggestion in suggestions:
        try:
            # Run the suggestion as a chat prompt
            response = agent.chat(f'Plot the following graph: {suggestion}')
            
            # Display the generated plot in Streamlit
            if response:
                st.pyplot(plt.gcf())  # Display the current figure
                plt.clf()  # Clear figure after each plot to avoid overlap
            else:
                st.write(f"Failed to generate plot for: {suggestion}")
        except Exception as e:
            st.write(f"Error generating plot for: {suggestion}. Error: {e}")

def query_based_analysis(agent,user_query):
    """
    Perform query-based analysis using a PandasAI agent.
    
    Args:
        df: The uploaded DataFrame.
        agent: Initialized PandasAI Agent.
    """
    if user_query:
        try:
            # Get the response from the agent
            response = agent.chat(user_query)
            
            # Check the type of response and display accordingly
            if isinstance(response, str) or isinstance(response, (int, float)):
                st.write("Response:", response)
            elif isinstance(response, plt.Figure):
                st.pyplot(response)
            else:
                st.write("Unsupported response type:", type(response))
            
            user_query = st.text_input("Enter your query:")
            query_based_analysis(agent, user_query) 
        except Exception as e:
            st.error(f"Error processing the query: {e}")

import matplotlib.pyplot as plt
import pandas as pd

def handle_multiple_queries(agent, queries):
    """
    Executes multiple queries using the PandasAI agent and displays results.
    Args:
        agent: The PandasAI agent initialized with the DataFrame.
        queries: A list of strings, where each string is a query.
    """
    for query in queries:
        st.write(f"**Query:** {query}")
        try:
            response = agent.chat(query)  # Get response from PandasAI
            
            if isinstance(response, str) or isinstance(response, (int, float)):
                # If the response is a string, integer, or float, display it directly
                st.write("Response:", response)
            
            elif isinstance(response, pd.DataFrame):
                # If the response is a DataFrame, display it as a table
                st.write("Response as DataFrame:")
                st.dataframe(response)
            
            elif isinstance(response, plt.Figure):
                # If the response is a matplotlib figure (plot), display it
                st.write("Generated Plot:")
                
                st.pyplot(plt.gcf())
                plt.clf()
            elif isinstance(response, str) and response.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.PNG')):
                # If the response is a file path to an image
                st.write(f"Displaying image from path: {response}")
                # Use Streamlit's st.image to display the image
                image = Image.open(response)  # Open the image
                st.image(image, caption="Generated Plot", use_column_width=True)
            
            else:
                st.write("Response type not supported:", type(response))

        except Exception as e:
            st.error(f"Error while processing query '{query}': {e}")



st.set_page_config(page_title="Automated Data Analysis", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Analysis & Report Generation", "Query Handling","ML Model"])

if page == "Analysis & Report Generation":
    # Streamlit application
    st.title("Analysis & Report Generation")

    # File upload button
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])
    flag=0
    if uploaded_file is not None:
        try:
            # Read the file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("File successfully uploaded!")
            
            # Display the first few rows of the file
            st.write("Preview of the uploaded file:")
            st.dataframe(df.head())

            # Get column names for identifier detection
            columns_of_dataset = df.columns.tolist()
            
            # Use the AI model to detect the identifier column
            st.info("Detecting identifier column using AI model...")
            identifier_index = get_identifier_column(columns_of_dataset)
            st.success(f"Identifier column detected: {columns_of_dataset[identifier_index]} (Column Index: {identifier_index})")
            
            # Analyze datatypes with the identifier column correction
            datatype_df = analyze_datatypes(df, identifier_index)

            # Display datatype analysis
            st.write("Datatype analysis of the uploaded file:")
            st.dataframe(datatype_df)

            print(datatype_df)
            print(type(datatype_df))
        
            # Perform data cleaning (missing values, duplicates)
            st.info("Performing data cleaning...")
            cleaned_df = data_cleaning(df, datatype_df)
            st.success("Data cleaning completed!")
            print(cleaned_df.head())

            df=cleaned_df
            st.dataframe(df)
            # Display the first few rows of the cleaned file
            st.write("Preview of the cleaned file:")
            st.dataframe(cleaned_df.head())

            # summary stats
            summarize_and_visualize(cleaned_df, datatype_df)

            #heatmap of correlation
            st.write("### Correlation Heatmap")
            draw_correlation_heatmap(cleaned_df, datatype_df)

            # Report and Query Section
  
            # Proceed with further processing only if input is provided
            import os
            os.environ["PANDASAI_API_KEY"] = "$2a$10$zpjvO2MZdia7AxQvPcrpUOlOBno3xLXdp6SCzwVF/y5zn1MDYYZYm"

            # Initialize Groq client and PandasAI Agent
            client = Groq(api_key="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")
            from pandasai import Agent
            agent = Agent(df)

                       
            # Call functions to handle the queries
            st.write('Plots suggested by AI')
            generate_report_with_pandasai(df, agent, client)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload a file to analyze.")


if page == "Query Handling":
    st.title('Query Handling')
    # File upload button
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])
    flag=0
    if uploaded_file is not None:
        try:
            # Read the file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("File successfully uploaded!")
            
            # Display the first few rows of the file

            # Get column names for identifier detection
            columns_of_dataset = df.columns.tolist()
            
            # Use the AI model to detect the identifier column
            identifier_index = get_identifier_column(columns_of_dataset)
            
            # Analyze datatypes with the identifier column correction
            datatype_df = analyze_datatypes(df, identifier_index)

            # Display datatype analysis

            print(datatype_df)
            print(type(datatype_df))
        
            # Perform data cleaning (missing values, duplicates)
            cleaned_df = data_cleaning(df, datatype_df)

            df=cleaned_df

            st.dataframe(df.head())

            # Report and Query Section
            st.write("Enter queries below")
            import os
            os.environ["PANDASAI_API_KEY"] = "$2a$10$zpjvO2MZdia7AxQvPcrpUOlOBno3xLXdp6SCzwVF/y5zn1MDYYZYm"          
            from pandasai import Agent
            agent = Agent(df)
           
                   
            # Text input for user query
            user_query = st.text_input("Enter your query:")

            if user_query:
                # Get the response from the agent
                response = agent.chat(user_query)
                
                # Check the type of response and display accordingly
                if isinstance(response, str) or isinstance(response, (int, float)):
                    st.write("Response:", response)
                elif isinstance(response, plt.Figure):
                    st.pyplot(response)
                else:
                    st.write("Unsupported response type:", type(response))
                
                # Continue the conversation
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload a file to analyze.")



# Page 4: ML Model (leave empty for now)
if page == "ML Model":
    st.title("Excel/CSV Datatype Analyzer with Identifier Detection")

    # File upload button
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read the file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("File successfully uploaded!")
            
            # Display the first few rows of the file
            st.write("Preview of the uploaded file:")
            st.dataframe(df.head())

            # Get column names for identifier detection
            columns_of_dataset = df.columns.tolist()
            
            # Use the AI model to detect the identifier column
            st.info("Detecting identifier column using AI model...")
            identifier_index = get_identifier_column(columns_of_dataset)
            st.success(f"Identifier column detected: {columns_of_dataset[identifier_index]} (Column Index: {identifier_index})")
            
            # Analyze datatypes with the identifier column correction
            datatype_df = analyze_datatypes(df, identifier_index)

            # Display datatype analysis
            st.write("Datatype analysis of the uploaded file:")
            st.dataframe(datatype_df)

            # Perform data cleaning (missing values, duplicates)
            st.info("Performing data cleaning...")
            cleaned_df = data_cleaning(df, datatype_df)
            st.success("Data cleaning completed!")

            df = cleaned_df
            from pycaret.classification import (
                setup as setup_clf,
                compare_models as compare_models_clf,
                pull as pull_clf,
                plot_model as plot_clf_model
            )
            from pycaret.regression import (
                setup as setup_reg,
                compare_models as compare_models_reg,
                pull as pull_reg,
                plot_model as plot_reg_model
            )

            ml_df = df.drop(df.columns[identifier_index], axis=1)
            st.write("The available columns are:", ml_df.columns.tolist())

            output_var = st.text_input("Enter the name of the required output variable:")

            if output_var in df.columns:
                try:
                    # Determine task type
                    if df[output_var].dtype in ['int64', 'float64'] and df[output_var].nunique() > 10:
                        task_type = "regression"
                    else:
                        task_type = "classification"

                    st.write(f"Detected task type: {task_type}")

                    if task_type == "classification":
                        st.write("Starting AutoML for Classification with PyCaret...")
                        with st.spinner("Running AutoML..."):
                            clf_setup = setup_clf(data=df, target=output_var)
                            best_model = compare_models_clf()
                            
                            # Pull the leaderboard
                            leaderboard = pull_clf()

                    elif task_type == "regression":
                        st.write("Starting AutoML for Regression with PyCaret...")
                        with st.spinner("Running AutoML..."):
                            reg_setup = setup_reg(data=df, target=output_var)
                            best_model = compare_models_reg()
                            
                            # Pull the leaderboard
                            leaderboard = pull_reg()

                    # Display leaderboard
                    st.success("AutoML Completed!")
                    st.write("Leaderboard of Models:")
                    st.dataframe(leaderboard)
                    import matplotlib.pyplot as plt

                    # Display the best model's summary and plots
                    st.write("Best Model Summary and Plots:")
                    st.subheader(f"Best Model: {best_model}")

                    if task_type == "classification":
                        # Display the best classification model summary
                        st.write("Summary for the Best Classification Model:")
                        st.write(best_model)

                        # Plot model metrics
                        st.write("Confusion Matrix:")
                        img = plot_clf_model(
                            best_model, plot="confusion_matrix", save=True
                        )
                        st.image(img)
                        
                        st.write("AUC Curve:")
                        img = plot_clf_model(
                            best_model, plot="auc", save=True
                        )
                        st.image(img)
                        
                        st.write("Feature Importance:")
                        img = plot_clf_model(
                            best_model, plot="feature", save=True
                        )
                        st.image(img)

                    elif task_type == "regression":
                        # Display the best regression model summary
                        st.write("Summary for the Best Regression Model:")
                        st.write(best_model)

                        # Plot model metrics
                        st.write("Residuals Plot:")
                        img = plot_reg_model(
                            best_model, plot="residuals", save=True
                        )
                        st.image(img)
                        
                        st.write("Error Plot:")
                        img = plot_reg_model(
                            best_model, plot="error", save=True
                        )
                        st.image(img)
                    

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.error("The entered output variable is not in the available columns.")
        
        except Exception as e:
            st.error(f"Error processing the file: {e}")