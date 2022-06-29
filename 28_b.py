
#Dataset Conversion
# Numerical to Categorical
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('IRIS.csv')
rf= RandomForestClassifier()

df['sepal_length']=pd.cut(df['sepal_length'], 3, labels=['0', '1', '2'])
df['sepal_width']=pd.cut(df['sepal_width'], 3, labels=['0', '1', '2'])
df['petal_length']=pd.cut(df['petal_length'], 3, labels=['0', '1', '2'])
df['petal_width']=pd.cut(df['petal_width'], 3, labels=['0', '1', '2'])


df=pd.read_csv("IRIS.csv")
X = df.drop("species",axis=1)
Y= df["species"]

print(Y)
le=LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)

rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
print('Random Forest: ', accuracy_score(Y_test,y_pred))


#Categorical to Numerical
le = LabelEncoder()
le.fit(Y)
Y = le.transform(Y)



# #Dealing with missing values
# # 1. Use Drop (df.drop())
# # 2. Use Replace (df.replace("back", "DOS"))
# # 3. Fill NA ()
#
# df['Item_Weight'].fillna((df['Item_Weight'].mean()), inplace=True)  #Imputing Numerical Values
#
# df['Outlet_Size'].fillna(('Medium'), inplace=True)  #Imputing Categorical Values



#Oversampling & Under Sampling
from imblearn.over_sampling import RandomOverSampler   #Random OverSampling
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)

from imblearn.over_sampling import SMOTE     #Synthetic Minority Oversampling (Smote)
sms = SMOTE(random_state=0)
X, Y = sms.fit_resample(X,Y)

from imblearn.under_sampling import RandomUnderSampler    #Random UnderSampling
rus=RandomUnderSampler (random_state=0)
X, Y=rus.fit_resample(X,Y)



#Identifying Outliers by ploting

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['sepal_length'])
plt.show()


#Identifying Outliers using Interquantile Range
print(df['sepal_length'])
Q1 = df['sepal_length'].quantile(0.25)
Q3 = df['sepal_length'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(upper)
print(lower)

out1=df[df['sepal_length'] < lower].values
out2=df[df['sepal_length'] > upper].values

df['sepal_length'].replace(out1,lower,inplace=True)
df['sepal_length'].replace(out2,upper,inplace=True)

print(df['sepal_length'])


#Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logr=LogisticRegression
pca=PCA(n_components=2)

df=pd.read_csv("IRIS.csv")
X = df.drop("species",axis=1)
Y= df["species"]

pca.fit(X)
X=pca.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)



