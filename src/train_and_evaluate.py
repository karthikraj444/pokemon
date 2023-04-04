import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import get_data,read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import json
import numpy as np
import os
import mlflow
from urllib.parse import urlparse


def eval_metrics(actual,pred):
    #rmse=np.sqrt(mean_squared_error(actual,pred))
    #mae=mean_absolute_error(actual,pred)
    #r2=r2_score(actual,pred)
    #return rmse,mae,r2
    accuracy=accuracy_score(actual,pred)
    return accuracy


def train_and_evaluate(config_path):
    config=read_params(config_path)
    test_data_path=config["split_data"]["test_path"]
    train_data_path=config["split_data"]["train_path"]
    random_state=config["base"]["random_state"]
    model_dir=config["model_dir"]
    tol=config["estimators"]["LogisticRegression"]["params"]["tol"]
    C=config["estimators"]["LogisticRegression"]["params"]["C"]
    max_iter=config["estimators"]["LogisticRegression"]["params"]["max_iter"]
    target=[config["base"]["target_col"]]
    train=pd.read_csv(train_data_path,sep=",")
    test=pd.read_csv(test_data_path,sep=",")

    train_x=train.drop(target,axis=1)
    test_x=test.drop(target,axis=1)

    train_y=train[target]
    test_y=test[target]

    ######
     ####MLFlow

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        lr=LogisticRegression(tol=tol,C=C,max_iter=max_iter,random_state=random_state)
        lr.fit(train_x,train_y)

        predicted_qualities=lr.predict(test_x)

        (accuracy)=eval_metrics(test_y,predicted_qualities)
    #print("RMSE:%s", rmse)
    #print("MAE:%s", mae)
    #print("R2:%s", r2)
        mlflow.log_param("tol",tol)
        mlflow.log_param("C",C)
        mlflow.log_param("max_iter",max_iter)


        mlflow.log_metric("accuracy",accuracy)
        
           
   
        tracking_url_type_store=urlparse(mlflow.get_artifact_uri()).scheme
        print("yes")
        print(tracking_url_type_store) 
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr,"model",registered_model_name=mlflow_config["registered_model_name"])
        else:
             mlflow.sklearn.load_model(lr,"model")




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)