# DataSet- California_Housing_price.csv
#  Linear regression Algorithm
# Using-1] Simple Linear Regression



import pandas as pd  # library used for data manipulation and analyze
import numpy as np  # it used for numerical operation  array,matrix,linear algebra operation
import matplotlib.pyplot as plt #Plotting graph ,visulalization

# To implement Linear regression need following libraries of sklearn(sci-kit learn)
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("California_Housing_Prices.csv")
df

#displaying some initial rows of dataset
df.head()

df.describe()

# Select independent (X) and dependent (y) variables
X=df[['median_income']]
y=df['median_house_value']


# Splitting the data into training and test sets (80% train, 20% test)
#training test 80%
#testing data 0.2 means 20%

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



# Creating a Linear Regression model
model=LinearRegression()


# Training the model on the training data
model.fit(x_train,y_train)


# Predicting the house values for the test data
y_pred=model.predict(x_test)


#Calculating Mean squared erroe

mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mse
rmse

#plotting regression line

plt.scatter(x_test, y_test, color="blue", label="Actual values")
plt.plot(x_test,y_pred,color="red",label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("Madian Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()

# Display RMSE
print(f"Root Mean Squared Error (RMSE): {rmse}")



#Above graph show Positive correlation:As median income increases, house prices also rise
#Rmse =84209.01241414454 :have average predication error

#above code for simple Linear Regression (which means 1 feature is used ) to predict price
#To improve the accuracy, we can use to multiple linear regression, which uses multiple features from the dataset to make predictions.


