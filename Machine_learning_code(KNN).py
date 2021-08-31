# Machine Learning

df['SalStat']=df['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df['SalStat'])

new_data=pd.get_dummies(df,drop_first=True)

cols=list(new_data)
print(cols)

features=list(set(cols)-set(['SalStat']))

x=new_data[features].values
y=new_data['SalStat']

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#KNN
          
model=KNeighborsClassifier(n_neighbors=5)
model.fit(train_x,train_y)
 

#prediction
pred=model.predict(test_x)
print(pred)

#confusion matrix
cm=confusion_matrix(test_y,pred)
print(cm)

#accuracy
acc=accuracy_score(test_y,pred)
print(acc)

#misclassified samples
print("Misclassified samples %d"%(test_y!=pred).sum())

#KNN efficiency
for i in range(1,10):
    model=KNeighborsClassifier(n_neighbors=i)
    model=model.fit(train_x,train_y)
    pred_i=model.predict(test_x)
    acc_i=accuracy_score(test_y,pred)
    print(acc_i)
    print("Misclassified samples %d"%(test_y!=pred).sum())
    
