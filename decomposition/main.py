import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('data/orders.txt', delimiter='\t', encoding='latin', parse_dates=['orderdate'])

print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())

df_sales = df.groupby('orderdate').sum()['totalprice'].to_frame().sort_index()
# It was Series in the first place. To fit in the model for time series, we need to change it to dataframe.
# To fit in the model for time series, it has to be sorted by time.
# Now it has been transformed from multivariate data set to time series data.
print(df_sales)

# Decomposition
#df_sales.plot()
#plt.show()

result = seasonal_decompose(df_sales,model='additive', period = 365)
#result.plot()
#plt.show()
print(result.trend)
print(result.seasonal.var())
print(result.resid)
print(result.observed)