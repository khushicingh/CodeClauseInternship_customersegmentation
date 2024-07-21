import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('customer_data.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS customers
                 (CustomerID TEXT, Age INTEGER, AnnualIncome REAL, SpendingScore REAL)''')
conn.commit()
customer_data = pd.read_sql_query("SELECT * FROM customers", conn)
output = widgets.Output()
def add_customer_data(customer_id, age, annual_income, spending_score):
    global customer_data
    with output:
        try:
            validate_input(customer_id, age, annual_income, spending_score)
            new_data = pd.DataFrame([[customer_id, age, annual_income, spending_score]],
                                    columns=["CustomerID", "Age", "AnnualIncome", "SpendingScore"])
            customer_data = pd.concat([customer_data, new_data], ignore_index=True)
            cursor.execute("INSERT INTO customers VALUES (?, ?, ?, ?)",
                           (customer_id, age, annual_income, spending_score))
            conn.commit()
            display_customer_data()
            print("Customer added successfully.")
        except ValueError as e:
            print(f"Error: {e}")
def validate_input(customer_id, age, annual_income, spending_score):
    if not customer_id:
        raise ValueError("Customer ID cannot be empty.")
    if not (0 <= age <= 120):
        raise ValueError("Age must be between 0 and 120.")
    if annual_income < 0:
        raise ValueError("Annual Income must be non-negative.")
    if not (0 <= spending_score <= 100):
        raise ValueError("Spending Score must be between 0 and 100.")
def display_customer_data():
    clear_output(wait=True)
    display(input_widgets)
    display(action_widgets)
    display(output)
    with output:
        clear_output(wait=True)
        display(customer_data)

def perform_clustering(n_clusters):
    global customer_data
    with output:
        if not customer_data.empty:
            kmeans = KMeans(n_clusters=n_clusters)
            customer_data["Cluster"] = kmeans.fit_predict(customer_data[["Age", "AnnualIncome", "SpendingScore"]])
            display_customer_data()
            plot_clusters()
            evaluate_clustering()
        else:
            print("No customer data to cluster")

def plot_clusters():
    with output:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=customer_data, x="AnnualIncome", y="SpendingScore", hue="Cluster", palette="viridis")
        plt.title("Customer Segments")
        plt.xlabel("Annual Income")
        plt.ylabel("Spending Score")
        plt.legend(title="Cluster")
        plt.show()

def evaluate_clustering():
    with output:
        score = silhouette_score(customer_data[["Age", "AnnualIncome", "SpendingScore"]], customer_data["Cluster"])
        print(f"Silhouette Score: {score:.2f}")

def save_data(filepath):
    global customer_data
    with output:
        customer_data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

def load_data(filepath):
    global customer_data
    with output:
        try:
            new_data = pd.read_csv(filepath)
            new_data.to_sql('customers', conn, if_exists='append', index=False)
            customer_data = pd.read_sql_query("SELECT * FROM customers", conn)
            display_customer_data()
            print(f"Data loaded from {filepath}")
        except Exception as e:
            print(f"Error loading data: {e}")

customer_id_input = widgets.Text(description="Customer ID:")
age_input = widgets.IntText(description="Age:")
annual_income_input = widgets.FloatText(description="Annual Income:")
spending_score_input = widgets.FloatText(description="Spending Score:")
add_button = widgets.Button(description="Add Customer")

def on_add_button_clicked(b):
    add_customer_data(customer_id_input.value, age_input.value, 
                      annual_income_input.value, spending_score_input.value)

add_button.on_click(on_add_button_clicked)

n_clusters_slider = widgets.IntSlider(value=3, min=2, max=10, step=1, description='Clusters:')
cluster_button = widgets.Button(description="Perform Clustering")
save_button = widgets.Button(description="Save Data")
load_button = widgets.Button(description="Load Data")
file_path_input = widgets.Text(description="File Path:")

def on_cluster_button_clicked(b):
    perform_clustering(n_clusters_slider.value)

def on_save_button_clicked(b):
    save_data(file_path_input.value)

def on_load_button_clicked(b):
    load_data(file_path_input.value)

cluster_button.on_click(on_cluster_button_clicked)
save_button.on_click(on_save_button_clicked)
load_button.on_click(on_load_button_clicked)

input_widgets = widgets.VBox([customer_id_input, age_input, annual_income_input, spending_score_input, add_button])

action_widgets = widgets.VBox([n_clusters_slider, cluster_button, file_path_input, save_button, load_button])

display(input_widgets)
display(action_widgets)
display(output)
