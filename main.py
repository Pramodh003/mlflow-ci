import numpy as np
import pandas as pd
data = pd.read_csv("Churn_Modelling.csv")
#print(data.head())
def cleaning_data(data):
    data.dropna()
    data.drop(["RowNumber","CustomerId","Surname"],axis=1,inplace=True)
    cat_vars = ["Geography","Gender"]
    for var in cat_vars:
        cat_list="var"+"_"+var
        cat_list=pd.get_dummies(data[var],prefix=var)
        data1=data.join(cat_list)
        data=data1
        cat_vars=["Geography","Gender"]
        data_vars=data.columns.values.tolist()
        to_keep=[i for i in data_vars if i not in cat_vars]
        final_data=data[to_keep]
        return final_data
final_data=cleaning_data(data)


print(final_data.head())


def spliting_dataset(final_data):
    from sklearn.model_selection import train_test_split
    X = final_data.drop(["Exited"],axis=1)
    y = final_data["Exited"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.3,random_state=10)
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test=spliting_dataset(final_data)
print(X_train,X_test,y_train,y_test)

def creating_mode(X_train,y_train):
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    return model

model = creating_mode(X_train,y_train)
print(model)


def y_pred(model,X_test):
    y_pred=model.predict(X_test)
    return y_pred

y_pred = y_pred(model,X_test)
print(y_pred)

def y_pred_prob(model,X_test):
    y_pred_prob=model.predict_proba(X_test)
    return y_pred_prob

y_pred_prob=model.predict_proba(X_test)
print(y_pred_prob)

def get_metrics(y_test,y_pred,y_pred_prob):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    entropy = log_loss(y_test,y_pred_prob)
    return {"accuracy":round(accuracy,2),"precision":round(precision,2),"recall":round(recall,2),"entropy":round(entropy,2)}

experiment_name = "testing-rf"
run_name = "churn_pred"
run_metrics = get_metrics(y_test, y_pred, y_pred_prob)
print(run_metrics)

def create_experiment(experiment_name,run_name,run_metrics,model,run_params=None):
    import mlflow
    mlflow.set_experiment(experiment_name)
    
    
    with mlflow.start_run():
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param , run_params[param])
        for metric in run_metrics:
            mlflow.log_metric(metric , run_metrics[metric])
        mlflow.sklearn.log_model(model , "model")
        
    
        mlflow.set_tag("tag1","RandomForest")
        mlflow.set_tags({"tag2":"Randomized Search CV", "tag3":"Production"})
        
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

create_experiment(experiment_name,run_name,run_metrics,model)