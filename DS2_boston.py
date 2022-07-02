import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score



lr =LinearRegression()

df =pd.read_csv("./HousingData.csv")
# print(df.describe())

# to fill Nan spaces
for i in df.columns:
    df[i].fillna((df[i].mean()),inplace=True)
print(df.head(10))

x=df.drop(columns=["MEDV"],axis=1)
print("All features",x)

y=df["MEDV"]
print("medv",y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

train=lr.fit(x_train,y_train)

y_p =lr.predict(x_test)
# print(accuracy_score(y_test,y_p))
print(mean_squared_error(y_test,y_p))



