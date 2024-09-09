import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("audi.csv")

df=df.drop(columns=["index","href","MileageRank","PriceRank","PPYRank","Score"])

engine_column=df["Engine"]
df_without_engine=df.drop(columns=["Engine"])
df=pd.get_dummies(df_without_engine, columns=["Type","Transmission","Fuel"],drop_first=True).astype(int)
df["Engine"]=engine_column
df.head()

df["Engine"]=df["Engine"].str.replace("L","")

df["Engine"]=pd.to_numeric(df["Engine"])

df=df.rename(columns={"Price(£)": "price"})

df.head()

y=df[["price"]]
x=df.drop("price", axis=1)

#WITHOUT TRAIN TEST
lm=LinearRegression()
model=lm.fit(x,y)
model.predict([[2015,50000,100,2,3000,1,0,1.4]])
model.score(x,y)

#WITH TRAIN TEST
x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.7,random_state=17)

lm=LinearRegression()
model=lm.fit(x_train,y_train)
model.score(x_test,y_test)
model.predict([[2015,50000,100,2,3000,1,0,1.4]])
