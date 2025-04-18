
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\Asus\Desktop\Python CA\Most Runs.csv")

# Strip column names to remove leading/trailing spaces
data.columns = data.columns.str.strip()

# Check the column names
print("Columns in the dataset:\n", data.columns)

# Show first few rows to visually inspect column values
print("\nSample data:\n", data.head())

# Fill missing values (basic strategy)
fill_columns = ['Player', 'Mat', 'Inns', 'Runs', 'Ave', 'SR', '100', '50']
for col in fill_columns:
    if col in data.columns:
        if data[col].dtype == 'O':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

# Check for any remaining nulls
print("\nMissing Values After Cleaning:\n", data.isnull().sum())

# Feature Engineering
if all(col in data.columns for col in ['Runs', 'Inns', 'SR', 'Ave', '100', '50']):
    data['Runs_per_Inns'] = data['Runs'] / data['Inns']
    data['Strike_to_Avg'] = data['SR'] / data['Ave']
    data['Total_Centuries'] = data['100'] + (data['50'] * 0.5)

# Graph 1: Top 10 Run Scorers

top10 = data.sort_values(by='Runs', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Runs', y='Player', data=top10, hue='Player', palette='viridis', legend=False)
plt.title('Top 10 Run Scorers')
plt.xlabel('Runs')
plt.ylabel('Player')
plt.tight_layout()
plt.show()

# -----------------------------------------
# 2. Players with Most Centuries
top100 = data.sort_values(by='100', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='100', y='Player', hue='Player', data=top100, palette='rocket', legend=False)
plt.title('Top 10 Players with Most Centuries')
plt.xlabel('Centuries')
plt.ylabel('Player')
plt.tight_layout()
plt.show()


# Graph 3: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Graph 4: Histogram of Averages
if 'Ave' in data.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Ave'], bins=30, kde=True)
    plt.title('Distribution of Batting Averages')
    plt.xlabel('Average')
    plt.tight_layout()
    plt.show()


    
# 4. Most Matches Played
top_matches = data.sort_values(by='Mat', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Mat', y='Player', hue='Player', data=top_matches, palette='mako', legend=False)
plt.title('Top 10 Players by Matches Played')
plt.xlabel('Matches')
plt.ylabel('Player')
plt.tight_layout()
plt.show()

# -----------------------------------------
# 5. Highest Individual Scores
data['HS_numeric'] = data['HS'].astype(str).str.replace('*', '', regex=False)
data['HS_numeric'] = pd.to_numeric(data['HS_numeric'], errors='coerce')
top_hs = data.sort_values(by='HS_numeric', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='HS_numeric', y='Player', hue='Player', data=top_hs, palette='crest', legend=False)
plt.title('Top 10 Highest Individual Scores')
plt.xlabel('Highest Score')
plt.ylabel('Player')
plt.tight_layout()
plt.show()

# ------------------- PIE CHART -------------------
top5 = data.sort_values(by='Runs', ascending=False).head(5)
plt.figure(figsize=(8, 8))
plt.pie(top5['Runs'], labels=top5['Player'], autopct='%1.1f%%', startangle=140)
plt.title('Top 5 Run Scorers Contribution')
plt.axis('equal')
plt.tight_layout()
plt.show()




# Modeling - Predict Ave from Runs
if 'Ave' in data.columns and 'Runs' in data.columns:
    X = data[['Runs']]
    y = data['Ave']

    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\nModel Coefficients:\nSlope: {model.coef_[0][0]:.2f}, Intercept: {model.intercept_[0]:.2f}")

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred_test):.4f}")

    # Plot regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled, y_scaled, alpha=0.5, label='Data')
    plt.plot(X_scaled, model.predict(X_scaled), color='red', linewidth=2, label='Regression Line')
    plt.title('Regression: Runs vs Average')
    plt.xlabel('Normalized Runs')
    plt.ylabel('Normalized Average')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Predict function
    def predict_average(runs):
        runs_scaled = scaler_X.transform([[runs]])
        avg_scaled = model.predict(runs_scaled)
        avg_original = scaler_y.inverse_transform(avg_scaled)[0][0]
        return round(avg_original, 2)

    # Example prediction
    example_runs = 10500
    print(f"\nPredicted Average for {example_runs} runs: {predict_average(example_runs)}")

    # Optional: User input
    try:
        user_input = float(input("Enter runs to predict average: "))
        print(f"Predicted Batting Average: {predict_average(user_input)}")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
else:
    print("Required columns ('Ave', 'Runs') not found in dataset.")
