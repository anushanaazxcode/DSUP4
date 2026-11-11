import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv("student_data.csv")
X=df[["study_hours","attendance"]]
y=df["passed"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
preds=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,preds))
print(classification_report(y_test,preds))
errors=X_test[preds!=y_test]
print("\nMisclassified Samples:")
print(errors)
