from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, ClientSecretCredential
import os
from config.auth import service_principal as creds
from config.auth import get_user_client_identity
from foodrating import foodrating
import mlflow
import mlflow.sklearn
from azure.ai.ml.entities import Environment, BuildContext, AmlCompute, ManagedIdentityConfiguration, IdentityConfiguration
import logging

CONFIG_FILE     = "config.json"
DATA_ASSET_NAME = "food-rating"
ML_ENVIRONMENT = "saena-ml-env"
ML_ENVIRONMENT_VERSION = "1"
ML_COMPUTE = "saena-ml-compute"
ML_COMPUTE_VERSION = "1"

def azure_ml_train_job():
     #Azure ML Studio connect
    #Connect to Azure with credentials. 
    print("Connecting to Azure workspace....")
    credentials = ClientSecretCredential(tenant_id=creds["tenantId"], 
                                         client_id=creds["clientId"], 
                                         client_secret=creds["clientSecret"], 
                                         )
    
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
    print(f"Path for data asset is {data.path}")
    print("Start training job......")
    # foodrating(data_asset_path=data.path)
    
    # Create  or retrieve Azure environment
    ml_env_name = ML_ENVIRONMENT
    ml_env_version = ML_ENVIRONMENT_VERSION
    try:
        print(f"Retrieving environment {ml_env_name}, version {ml_env_version} from Azure ML Studio.......")
        ml_client_environment=mlclient.data.get(name=ml_env_name, version=ml_env_version)
    except Exception as e:
        print(f"Environment {ml_env_name}, version {ml_env_version} does not exist. Attempting to create....")
        print("Error raised is ", e.__cause__)
        ml_env_description = "Environment to run food rating experiments"
        ml_env_docker_build_context = BuildContext(path=os.path.join("azure", "environment"), dockerfile_path="Dockerfile") 
        ml_client_environment_create = Environment(
            name = ml_env_name, 
            version = ml_env_version, 
            build = ml_env_docker_build_context, 
            description = ml_env_description, 
            tags=dict(org="saena", sub_org="ml"), 
            properties=dict(type="regression"), 
            datastore="workspaceblobstore"
        )
        print("Creating environment in Azure ML Studio ....")
        ml_client_environment = mlclient.environments.create_or_update(ml_client_environment_create)
    finally:
         print(f"Retrieved ML Client Environment with resource id {ml_client_environment.id} and version {ml_client_environment.version}")
        
    # Create or retrieve compute environment
    print("Retrieving ML Compute Instance ", ML_COMPUTE)
    try:
        mlclient.compute.get(name=ML_COMPUTE)
    except Exception:
        print(f"Compute environment does not exist. Attempting to create ....")
        # Retrieve user managed identity, and assign the identity to the compute environment
        user_managed_id = get_user_client_identity()
        user_managed_id_config_mng = ManagedIdentityConfiguration(
            # client_id=user_managed_id["clientId"],
            # principal_id=user_managed_id["principalId"],
            resource_id=user_managed_id["id"]
                                      
                                                              )
        user_managed_id_config = IdentityConfiguration(user_assigned_identities=[user_managed_id_config_mng],type="UserAssigned")
        ml_compute = AmlCompute(name=ML_COMPUTE, 
                                description="ML Compute intance for food rating and other services",
                                tags=dict(org="saena", sub_org="ml"),
                                identity=user_managed_id_config,
                                min_instances=0,
                                max_instances=4
                                )
        mlclient.compute.begin_create_or_update(ml_compute).wait()
    finally: 
        print(f"Compute Instance Created...")
    
    #Create command to execute
    ml_command_job = command(name="food2", 
                             description="food rating training",
                             command="python foodrating.py --registered-model-name ${{inputs.registered_model_name}} --data ${{inputs.data}}",
                             inputs=dict(
                                 data=Input(type="uri_file", 
                                            path="https://saenamlservice7088340037.blob.core.windows.net/azureml-blobstore-f9c0608f-1990-4c29-b590-4c5e114bb0c1/UI/2024-09-27_064707_UTC/zomato.csv"),
                                 registered_model_name="model"
                             ), 
                             #compute=ML_COMPUTE, 
                             environment=ml_client_environment
                             )
    try: 
        mlclient.jobs.create_or_update(job=ml_command_job)
    except Exception as hp:
        print("Error Occured", hp.reason, hp.model, hp.response, hp.response, hp.error)
        raise hp
    
    