import pandas as pd
import numpy as np 
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import os

plot_directory = 'regression'

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
    print(f"Directory '{plot_directory}' was created.")
else:
    print(f"Directory '{plot_directory}' already exists.")

print("Loading dataset...")
file_path = 'regressionDataset/car_prices.csv'
df = pd.read_csv(file_path)
print("Dataset loaded. Here are the first few rows:")
print(df.head())

print("\nInitial Data Exploration")
print("Dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())


### Data Cleaning ###
print("\nStarting Data Cleaning")
df = df.drop(columns=['vin'])
print("'vin' column dropped.")

df = df.dropna(subset=['sellingprice'])
print("Rows with missing 'sellingprice' dropped.")

df['condition'] = df['condition'].fillna(df['condition'].median())
df['odometer'] = df['odometer'].fillna(df['odometer'].median())
print("Missing 'condition' and 'odometer' values imputed with median.")

categorical_columns = df.select_dtypes(include=['object']).columns
cardinality = df[categorical_columns].nunique()
print(cardinality)

### ANOVA for Feature Relevance ###
print("\nAssessing Feature Relevance with ANOVA")
anova_results = {}
for feature in ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']:
    groups = [group['sellingprice'].dropna().values for name, group in df.groupby(feature) if group['sellingprice'].dropna().shape[0] > 1]
    if groups:  # Check if there are any groups to compare
        f_statistic, p_value = stats.f_oneway(*groups)
        anova_results[feature] = (f_statistic, p_value)
for feature, (f_statistic, p_value) in anova_results.items():
    print(f"{feature}: F-statistic = {f_statistic}, p-value = {p_value}")


### Further Data Cleaning ###
    
print("\nFurther Data Cleaning")
df = df.drop(['saledate', 'seller', 'model', 'trim'], axis=1)
print("Dropped 'saledate', 'seller', 'model', 'trim'.")
df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
df = df.dropna()
print("Dropped any remaining rows with NaN values.")

# Encoding
print("\nEncoding categorical variables...")
df_encoded = pd.get_dummies(df)
print("Encoding complete.")

if df_encoded.isnull().any().any():
    print("Warning: There are still NaN values in the DataFrame after encoding.")
else:
    print("No NaN values in the DataFrame after encoding. Proceeding to model training.")

### Correlation Analysis ###
    
print("\nConducting Correlation Analysis")
dfNumeric = df_encoded[['year', 'condition', 'odometer', 'mmr', 'sellingprice']]
correlation_matrix_numeric = dfNumeric.corr()
correlation_matrix_numeric[np.abs(correlation_matrix_numeric) < .2] = 0
plt.figure(figsize=(5, 5))
sns.heatmap(correlation_matrix_numeric, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.savefig('regression/numeric_features_correlation.png')
print("Saved numeric features correlation heatmap.")

dfOHE = df_encoded.drop(['year', 'condition', 'odometer', 'mmr'], axis=1)
correlation_matrix_ohe = dfOHE.corr()
correlation_matrix_ohe[np.abs(correlation_matrix_ohe) < .2] = 0
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_matrix_ohe[['sellingprice']].sort_values(by=['sellingprice'], ascending=False).head(50), vmin=-1, cmap='viridis', annot=True)
plt.title('Top 50 One-Hot Encoded Features Correlation with Selling Price')
plt.savefig('regression/ohe_features_correlation.png')
print("Saved one-hot encoded features correlation heatmap.")

print("\nPreparing Data for Modeling")
X = dfNumeric.drop(['sellingprice'], axis=1).values
y = dfNumeric[['sellingprice']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("Data split into training and test sets.")

print("\nPreprocessing complete. Ready for model training.")

### Regrassion ###

# Updated Model Training and Evaluation Function
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, poly=False):
    if poly:
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        model.fit(X_train_poly, y_train.ravel())  # Fit model to polynomial features
    else:
        model.fit(X_train, y_train.ravel())
    
    y_pred = model.predict(X_test_poly if poly else X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Coefficient of Determination (R^2):", r2)
    
    if hasattr(model, 'coef_'):
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
    
    return mse, mae, r2, y_pred

models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = {}
for name, model in models.items():
    poly = True if name == "Polynomial Regression" else False
    results[name] = train_evaluate_model(model, X_train, y_train, X_test, y_test, name, poly=poly)

metrics = ['MSE', 'R^2', 'MAE']
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Regression Model Performance Comparison')

for i, metric in enumerate(metrics):
    values = [results[model][i] for model in models]
    axs[i].bar(models.keys(), values, color=['blue', 'orange', 'green', 'red'])
    axs[i].set_title(metric)
    axs[i].set_ylabel(metric)
    axs[i].set_xticklabels(models.keys(), rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

### Plotting Regression Line Using MMR ###
mmr_test = X_test[:, 3] 

def plot_predictions(mmr_test, y_test, y_preds, model_name, show=True, save=False):
    plt.figure(figsize=(12, 8))
    plt.scatter(mmr_test, y_test, color='gray', alpha=0.5, label='Actual Selling Prices')
    plt.scatter(mmr_test, y_preds, alpha=0.5, label=f'{model_name} Predictions')
    plt.xlabel('MMR (Manheim Market Report value)')
    plt.ylabel('Selling Price')
    plt.title(f'Actual vs. Predicted Selling Prices - {model_name}')
    plt.legend()
    if save:
        plt.savefig(f'regression/{model_name.replace(" ", "_")}_predictions.png')
    if show:
        plt.show()
    plt.close()

for name, result in results.items():
    _, _, _, y_pred = result
    plot_predictions(mmr_test, y_test, y_pred, name, show=False, save=True)

plt.figure(figsize=(12, 8))
plt.scatter(mmr_test, y_test, color='gray', alpha=0.5, label='Actual Selling Prices')
colors = ['blue', 'green', 'red', 'purple']  # Colors for different models
for i, (name, result) in enumerate(results.items()):
    _, _, _, y_pred = result
    plt.scatter(mmr_test, y_pred, alpha=0.5, color=colors[i], label=f'{name} Predictions')
plt.xlabel('MMR (Manheim Market Report value)')
plt.ylabel('Selling Price')
plt.title('Actual vs. Predicted Selling Prices for All Models')
plt.legend()
plt.savefig('regression/all_models_predictions.png')
plt.show()