# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
#----------------------------------------

# Our data
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Getting the total production per yer
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Creating the variable X, which contains the column of year from the prod_per_year dataframe
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Creating the variable y, which contains the column totalprod from the prod_per_year dataframe
y =  prod_per_year['totalprod']

# Now let's see the correlation between this two features. Plotting the features in a scatterplot graph
plt.scatter(X, y)
plt.show()

# Ok, now we can create and fit a linear regression model. First, lets instantiate our linear regression model
regr = linear_model.LinearRegression()

# Fit the model with the correct data
regr.fit(X, y)

# Lets see the slope and the intercept
#print(regr.coef_)
#print(regr.intercept_)

# Ok! So now we can create a list of predictions for y
y_predict = regr.predict(X)

# Lets plot this values to see how its working
plt.plot(X, y_predict)
plt.show()

# Just with this data we can easily see the decline in honey production through the years. But we want more, we want to see the future!!! Yeah, lets predict the honey decline
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1,1)

# Now lets call our Magic Crystal Ball!
future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()