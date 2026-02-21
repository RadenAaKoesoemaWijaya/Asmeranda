@echo off
REM Azure Deployment Script for Asmeranda ML Application (Windows)
REM This script automates the deployment to Azure Container Apps

setlocal enabledelayedexpansion

REM Configuration
set RESOURCE_GROUP=asmeranda-ml-rg
set LOCATION=eastus
set ACR_NAME=asmerandacr%date:~-4,4%%date:~-10,2%%date:~-7,2%%time:~0,2%%time:~3,2%%time:~6,2%
set CONTAINER_APP_NAME=asmeranda-ml-app
set CONTAINER_ENV_NAME=asmeranda-container-env
set KEY_VAULT_NAME=asmeranda-kv%date:~-4,4%%date:~-10,2%%date:~-7,2%%time:~0,2%%time:~3,2%%time:~6,2%
set STORAGE_ACCOUNT=asmerandasa%date:~-4,4%%date:~-10,2%%date:~-7,2%%time:~0,2%%time:~3,2%%time:~6,2%

echo Starting Azure deployment for Asmeranda ML Application...

REM Check if Azure CLI is installed
where az >nul 2>nul
if %errorlevel% neq 0 (
    echo Azure CLI is not installed. Please install it first.
    exit /b 1
)

REM Login to Azure
echo Logging in to Azure...
az login
az account show

REM Create resource group
echo Creating resource group: %RESOURCE_GROUP%...
az group create --name %RESOURCE_GROUP% --location %LOCATION%

REM Create Azure Container Registry
echo Creating Azure Container Registry: %ACR_NAME%...
az acr create --resource-group %RESOURCE_GROUP% --name %ACR_NAME% --sku Basic --location %LOCATION%
az acr login --name %ACR_NAME%

REM Build and push Docker image
echo Building and pushing Docker image...
docker build -f Dockerfile.azure -t %ACR_NAME%.azurecr.io/asmeranda-ml-app:latest .
docker push %ACR_NAME%.azurecr.io/asmeranda-ml-app:latest

REM Create Key Vault
echo Creating Key Vault: %KEY_VAULT_NAME%...
az keyvault create --resource-group %RESOURCE_GROUP% --name %KEY_VAULT_NAME% --location %LOCATION%
az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "super-admin-password" --value "Admin@12345"
az keyvault secret set --vault-name %KEY_VAULT_NAME% --name "smtp-password" --value "your-smtp-password"

REM Create storage account
echo Creating Storage Account: %STORAGE_ACCOUNT%...
az storage account create --resource-group %RESOURCE_GROUP% --name %STORAGE_ACCOUNT% --location %LOCATION% --sku Standard_LRS

REM Create Container Apps environment
echo Creating Container Apps Environment...
az containerapp env create --name %CONTAINER_ENV_NAME% --resource-group %RESOURCE_GROUP% --location %LOCATION%

REM Deploy Container App using ARM template
echo Deploying Container App...
az deployment group create --resource-group %RESOURCE_GROUP% --template-file azure-infrastructure.json --parameters appName=%CONTAINER_APP_NAME% environmentName=%CONTAINER_ENV_NAME% acrName=%ACR_NAME% keyVaultName=%KEY_VAULT_NAME% storageAccountName=%STORAGE_ACCOUNT% location=%LOCATION%

REM Setup monitoring
echo Setting up monitoring...
az monitor app-insights component create --app asmeranda-app-insights --location %LOCATION% --resource-group %RESOURCE_GROUP%

REM Get application URL
echo Getting application URL...
for /f "tokens=*" %%i in ('az containerapp show --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --query properties.configuration.ingress.fqdn -o tsv') do set APP_URL=%%i
echo Application URL: https://%APP_URL%

echo Deployment completed successfully!
echo Your application is now running on Azure Container Apps
echo Remember to configure your SMTP settings in the application

pause