# Machine Learning

df['SalStat']=df['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
           
print(df['SalStat'])

new_data=pd.get_dummies(df,drop_first=True)

col_list=list(new_data)
print(col_list)

features=list(set(col_list)-set(['SalStat']))
print(features)

y=df['SalStat'].values
print(y)

x=new_data[features].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#model
model=LogisticRegression()

model.fit(train_x,train_y)

print(model.coef_)
print(model.intercept_)

#prediction

pred=model.predict(test_x)
print(pred)

#confusion matrix

cm=confusion_matrix(test_y,pred)
print(cm)

#accuracy
acc=accuracy_score(test_y,pred)
print(acc)

