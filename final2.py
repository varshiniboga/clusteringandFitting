import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import numpy as np 
from scipy import stats
from sklearn.metrics import silhouette_score

def read_clean_transpose_data(data_path):
    """
    Read, clean, and transpose the data.

    Parameters:
    - data_path (str): File path of the dataset.

    Returns:
    - original_data (pd.DataFrame): Original data read from the file.
    - cleaned_data (pd.DataFrame): Data after handling NaN values and converting to numeric.
    - transposed_data (pd.DataFrame): Transposed data.
    """
    # Read the original dataset
    original_data = pd.read_csv(data_path)

    # Create a copy for cleaning
    cleaned_data = original_data.copy()

    # Extract relevant columns
    columns_of_interest = ['Urban population (% of total population) [SP.URB.TOTL.IN.ZS]' ,
                            'Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]' ,
                            'Rural population (% of total population) [SP.RUR.TOTL.ZS]' ,
                            'Population in urban agglomerations of more than 1 million (% of total population) [EN.URB.MCTY.TL.ZS]']

    cleaned_data[columns_of_interest] = cleaned_data[columns_of_interest].replace('..' , np.nan).astype(float)

    # Replace NaN values with the mean of each column
    for column in cleaned_data.columns:
        if cleaned_data[column].dtype == 'object':
            cleaned_data[column] = pd.to_numeric(cleaned_data[column] , errors = 'coerce')

    # Fill NaN values with the mean of each column
    cleaned_data.fillna(cleaned_data.mean() , inplace = True)

    # Transpose the data
    transposed_data = cleaned_data.transpose()

    return original_data , cleaned_data , transposed_data

def polyFunc(x, *coeffs):
    """
    Polynomial model function.

    Parameters:
    - x (array-like): Input values.
    - coeffs (float): Coefficients of the polynomial.

    Returns:
    - array-like: Output values based on the polynomial model.
    """
    x_array = np.asarray(x, dtype=float)
    return np.polyval(coeffs, x_array)

def conf_interval_poly(params, covariance, x_values, confidence=0.95):
    """
    Calculate confidence intervals for predicted values using a polynomial model.

    Parameters:
    - params (array-like): Coefficients of the polynomial model.
    - covariance (array-like): Covariance matrix of the polynomial model.
    - x_values (array-like): Input values for which confidence intervals are calculated.
    - confidence (float, optional): Confidence level for the interval (default is 0.95).

    Returns:
    - tuple: Lower and upper bounds of the confidence intervals.
    """

    alpha = 1 - confidence
    n = len(x_values)
    dof = max(0, n - len(params))
    t_value = stats.t.ppf(1 - alpha / 2, dof)
    error = np.sqrt(np.diag(covariance))

    lower_bound = np.zeros(n)
    upper_bound = np.zeros(n)

    for i in range(min(n, len(params))):
        lower_bound[i] = polyFunc(x_values[i], *params) - t_value * error[i]
        upper_bound[i] = polyFunc(x_values[i], *params) + t_value * error[i]

    return lower_bound, upper_bound

data_path = '70216183-4379-43c5-bc1e-929b6a85aa40_Data.csv'
original_data, cleaned_data, transposed_data = read_clean_transpose_data(data_path)

columns_of_interest = ['Urban population (% of total population) [SP.URB.TOTL.IN.ZS]',
                        'Total natural resources rents (% of GDP) [NY.GDP.TOTL.RT.ZS]',
                        'Rural population (% of total population) [SP.RUR.TOTL.ZS]',
                        'Population in urban agglomerations of more than 1 million (% of total population) [EN.URB.MCTY.TL.ZS]']

X = cleaned_data[columns_of_interest]
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)
cleaned_data['Cluster'] = kmeans.fit_predict(X_normalized)
silhouette_avg = silhouette_score(X_normalized, cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

cluster_centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(cleaned_data[columns_of_interest[0]], cleaned_data[columns_of_interest[1]],
            c=cleaned_data['Cluster'], cmap='viridis')
plt.scatter(cluster_centers_original_scale[:, 0], cluster_centers_original_scale[:, 1], marker='X',
            s=200, color='red', label='Cluster Centers')
plt.xlabel(columns_of_interest[0])
plt.ylabel(columns_of_interest[1])
plt.title('KMeans Clustering')
plt.legend()
plt.show()

years = original_data['Time']
urban_population = original_data['Population in urban agglomerations of more than 1 million (% of total population) [EN.URB.MCTY.TL.ZS]']

mask = np.isfinite(urban_population)
urban_population = urban_population[mask]
years = years[mask]

years_str = years.astype(str)
urban_population = urban_population.values
years_str = years_str.values

degree = 2
params, covariance = np.polyfit(years_str.astype(float), urban_population, degree, cov=True)

future_years = np.array([2020, 2030, 2040])
predicted_values_future = polyFunc(future_years.astype(float), *params)
print(future_years)
print(predicted_values_future)

all_years = np.concatenate([years_str, future_years])
predicted_values_all = polyFunc(all_years.astype(float), *params)

lower_bound, upper_bound = conf_interval_poly(params, covariance, all_years)

plt.scatter(years_str, urban_population, label='Actual Data', color='pink')
plt.plot(all_years.astype(str), predicted_values_all,
         label='Predicted Values', color='orange')
plt.fill_between(all_years.astype(str), lower_bound, upper_bound,
                 color='black', alpha=0.2, label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Urban population (% of total population)')
plt.legend()
plt.show()
