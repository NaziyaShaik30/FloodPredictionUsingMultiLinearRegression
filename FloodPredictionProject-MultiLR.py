import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("Project_test.csv")
data1=pd.read_csv("Project_train.csv")
#data types of the features
print(data.info())
print(data1.info())
#mean median etc..
print(data.describe())
print(data1.describe())
#checking whether any null values present in the data
print(data.isnull())
#sum the all null values in the whole data
print(data.isnull().sum())
print(data1.isnull())
print(data1.isnull().sum())
#printing first 5 values of the data
print(data.head())
print(data1.head())
#heatmap
#correlation between the features and target
plt.figure(dpi=125)
sns.heatmap(np.round(data1.corr(numeric_only=True),2),annot=True)
plt.show()

x=data1[['MonsoonIntensity','TopographyDrainage','RiverManagement','Deforestation','Urbanization','ClimateChange','DamsQuality','Siltation','AgriculturalPractices','Encroachments','IneffectiveDisasterPreparedness','DrainageSystems','CoastalVulnerability','Landslides','Watersheds','DeterioratingInfrastructure','PopulationScore','WetlandLoss','InadequatePlanning','PoliticalFactors']]
y=data1['FloodProbability']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#model selection
model=LinearRegression()
#training or fitting the data
model.fit(x_train,y_train)

#predicting
y_pred=model.predict(x_test)

# acc=accuracy_score(y_test,y_pred)
# print(f"accuracy ",acc)

plt.figure(dpi=125)
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Regression Line for y_test vs y_pred')
plt.show()


plt.scatter(data1['RiverManagement'],data1['FloodProbability'])
plt.xlabel('RiverManagement')  # Label for x-axis
plt.ylabel('FloodProbability')  # Label for y-axis
plt.title('Scatter Plot of  RiverManagement vs FloodProbability  ')
plt.show()

plt.scatter(data1['Deforestation'],data1['FloodProbability'])
plt.xlabel('Deforestation')  # Label for x-axis
plt.ylabel('FloodProbability')  # Label for y-axis
plt.title('Scatter Plot of  Deforestation vs FloodProbability  ')
plt.show()
plt.scatter(data1['Urbanization'],data1['FloodProbability'])
plt.xlabel('Urbanization')  # Label for x-axis
plt.ylabel('FloodProbability')  # Label for y-axis
plt.title('Scatter Plot of  Urbanization vs FloodProbability  ')
plt.show()
plt.scatter(data1['ClimateChange'],data1['FloodProbability'])
plt.xlabel('ClimateChange')  # Label for x-axis
plt.ylabel('FloodProbability')  # Label for y-axis
plt.title('Scatter Plot of  ClimateChange vs FloodProbability  ')
plt.show()
plt.figure(figsize=(20, 15))

# Iterate over each feature and create a box plot
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"mean squared error",mse)
print(f"r^2 error",r2)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5, scoring='r2')
print(f"Cross-validated R² scores: {scores}")
print(f"Mean R² score: {np.mean(scores)}")
