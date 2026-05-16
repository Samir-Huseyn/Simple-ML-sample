import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("listings.csv")

print(df.info())
print(df.describe())
print(df.head())
print(df.shape)


print(df.isnull().sum())
print(df.duplicated().sum())

df = df.drop(columns=["id", "name", "host_id", "host_name", "license", "last_review", "neighbourhood"])
df = df.dropna(subset=["price"])
df["reviews_per_month"] = df["reviews_per_month"].fillna(0) 
df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis = 1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(mae, r2)


plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Prices')
plt.show()



plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation')
plt.show()