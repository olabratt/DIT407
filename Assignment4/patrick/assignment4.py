import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

SEED = 1234

df = pd.read_csv("Assignment4/life_expectancy.csv", encoding='utf-8')
df = df.drop('Country', axis=1)

leb_column = 'Life Expectancy at Birth, both sexes (years)'

## Problem 1

df_train, df_test = train_test_split(df, test_size=0.2 ,random_state=SEED)

## Problem 2

print("\n====== Problem 2 ======\n")
# max_coeff = 0
# max_column = ""

# for column in df_train.drop(leb_column, axis=1):
#     df_notna = df_train[df_train[column].notna()]
#     r = pearsonr(df_notna[leb_column], df_notna[column])[0]
#     if r > max_coeff:
#         max_coeff = r
#         max_column = column

# print("Variable with the strongest linear relationship :", max_column)
# print("Pearson correlation coefficient :", max_coeff)

hdi_column = 'Human Development Index (value)'
df_notna = df_train[df_train[hdi_column].notna()]
x = df_notna[[hdi_column]]
y = df_notna[leb_column]

model = LinearRegression().fit(x, y)
y_pred = model.predict(x)
print("Coefficient of determination :", r2_score(y, y_pred))
print("Slope of the line :", model.coef_)
print("Intercept of the line :", model.coef_)

# xfit = np.linspace(x.min()[0], x.max()[0], 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y, s=4, color = 'red')
# plt.plot(xfit, yfit, color = 'blue', linewidth = 2)
# plt.title("Linear regression between HDI and LEB")
# plt.xlabel("Human Development Index (HDI)")
# plt.ylabel("Life expectancy at birth (LEB) in years")
# plt.show()

df_notna = df_test[df_test[hdi_column].notna()]
x_test = df_notna[hdi_column]
y_test = df_notna[leb_column]
y_test_pred = model.predict(x_test[:, np.newaxis])
r = pearsonr(y_test, y_test_pred)[0]
mse = mean_squared_error(y_test, y_test_pred)
print("Pearson coefficient :", r)
print("Mean squarred error :", mse)

plt.scatter(x_test, y_test, s=4, color = 'red')
plt.plot(x_test, y_test_pred, color = 'blue', linewidth = 2)
plt.title("Prediction of LEB for the test set")
plt.xlabel("Human Development Index (HDI)")
plt.ylabel("Life expectancy at birth (LEB) in years")
plt.show()

# Problem 3
print("\n====== Problem 3 ======\n")
correlation = df_train.drop(hdi_column, axis = 1).corr(method='spearman')[leb_column].drop(leb_column)
max_corr_column = correlation.abs().idxmax()
max_corr = correlation.abs().max()
print("Maximum correlation variable : ", max_corr_column)
print("Spearman correlation coefficient :", max_corr)
df = df_train[df_train[max_corr_column].notna()]
p = pearsonr(df[max_corr_column], df[leb_column])[0]
print("Pearson correlation coefficient before applying log function :", p)
p = pearsonr(np.log(df[max_corr_column]), df[leb_column])[0]
print("Pearson correlation coefficient after applying log function :", p)

# Problem 4
print("\n====== Problem 4 ======\n")
df_train = df_train.drop(hdi_column, axis = 1)
correlation = df_train.corr(method='pearson')[leb_column].drop(leb_column)
max_corr_columns = [correlation.abs().idxmax()]
max_corr_columns.append(correlation.drop(max_corr_columns[0]).abs().idxmax())
print("Best variables :", max_corr_columns)
print("Pearson correlation coefficients :", correlation[max_corr_columns[0]], correlation[max_corr_columns[1]])
print(pearsonr([df_train[max_corr_columns[0]], df_train[max_corr_columns[1]]], df_train[leb_column]))