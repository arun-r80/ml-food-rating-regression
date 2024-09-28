from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import os
from foodrating import foodrating

CONFIG_FILE     = "config.json"
DATA_ASSET_NAME = "food-rating"

def azure_ml_train_job():
     #Azure ML Studio connect
    #Connect to Azure with credentials. 
    print("Connecting to Azure workspace....")
    try: 
        credentials = DefaultAzureCredential()
    except:
        credentials=InteractiveBrowserCredential()
    print("Azure Credentials secured....")
    print("Connect to Azure ML Studio")
    try:
        mlclient = MLClient.from_config(credential=credentials, path=os.path.join("config"),file_name=CONFIG_FILE )
    except Exception as e: 
        print("Connecection to Azure ML Studio failed...")
        raise e
    print("Connected to Azure ML Studio") 

    ##Get Data Asset
    print("Get Data Asset..........")
    data = mlclient.data.get(name=DATA_ASSET_NAME,version="1" )
    print("Retrieved Data Asset....")
    print("Start training job......")
    # foodrating(data_asset_path=data.path)
        
