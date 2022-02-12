import pandas
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
df = pandas.read_csv('C:\python1\project\project\cars.csv')
X=df[['Weight','Volume']]
y=df['CO2']
regr=LinearRegression()
regr.fit(X,y)
test_y=regr.predict(X)
predictedCO2=regr.predict([[2300,1300]])
print(predictedCO2)