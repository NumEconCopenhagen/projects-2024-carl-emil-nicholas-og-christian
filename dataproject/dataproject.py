# dataproject.py

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class DataPlotter:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.total_data_set_sorted = None

    def merge_and_prepare_data(self):
        # Merge the two datasets
        total_data_set = pd.concat([self.data1, self.data2])
        
        # Sort the Total_data_set by 'Year' in ascending order
        self.total_data_set_sorted = total_data_set.sort_values('Year')
    
    def print_data_summary(self):
        if self.total_data_set_sorted is None:
            raise ValueError("Data not prepared. Call merge_and_prepare_data() first.")
        
        # Print the first 10 observations
        print(self.total_data_set_sorted.head(10))
        
        # Print the descriptive statistics for 'Temp (celsius)' and 'Impact (%)'
        print(self.total_data_set_sorted[['Temp (celsius)', 'Impact (%)']].describe())

    def plot_data(self):
        if self.total_data_set_sorted is None:
            raise ValueError("Data not prepared. Call merge_and_prepare_data() first.")
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Temp (celsius)', y='Impact (%)', data=self.total_data_set_sorted)

        # Linear regression
        model_linear = LinearRegression()
        X = self.total_data_set_sorted['Temp (celsius)'].values.reshape(-1, 1)
        y = self.total_data_set_sorted['Impact (%)'].values
        model_linear.fit(X, y)
        sns.lineplot(x=self.total_data_set_sorted['Temp (celsius)'], y=model_linear.predict(X), color='red')

        # Quadratic regression
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model_quad = LinearRegression()
        model_quad.fit(X_poly, y)
        X_fit = np.linspace(self.total_data_set_sorted['Temp (celsius)'].min(), self.total_data_set_sorted['Temp (celsius)'].max(), 100)
        X_fit_poly = poly.fit_transform(X_fit.reshape(-1, 1))
        sns.lineplot(x=X_fit, y=model_quad.predict(X_fit_poly), color='green')

        plt.title('Impact vs. Temp')
        plt.xlabel('Temp (celsius)')
        plt.ylabel('Impact (%)')
        plt.legend(['Data Points', 'Linear Regression', 'Quadratic Regression'])
        plt.grid(True)
        plt.show()
