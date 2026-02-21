#!/bin/bash
# Azure Deployment Script for Asmeranda ML Application
# This script automates the deployment to Azure Container Apps

set -e

# Configuration
RESOURCE_GROUP="asmeranda-ml-rg"
LOCATION="eastus"
ACR_NAME="asmerandacr$(date +%s)"
CONTAINER_APP_NAME="asmeranda-ml-app"
CONTAINER_ENV_NAME="asmeranda-container-env"
KEY_VAULT_NAME="asmeranda-kv$(date +%s)"
STORAGE_ACCOUNT="asmerandasa$(date +%s)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Azure deployment for Asmeranda ML Application...${NC}"

# Function to check if Azure CLI is installed
check_azure_cli() {
    if ! command -v az &> /dev/null; then
        echo -e "${RED}Azure CLI is not installed. Please install it first.${NC}"
        exit 1
    fi
}

# Function to login to Azure
azure_login() {
    echo -e "${YELLOW}Logging in to Azure...${NC}"
    az login
    az account show
}

# Function to create resource group
create_resource_group() {
    echo -e "${YELLOW}Creating resource group: $RESOURCE_GROUP...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
}

# Function to create Azure Container Registry
create_acr() {
    echo -e "${YELLOW}Creating Azure Container Registry: $ACR_NAME...${NC}"
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --location $LOCATION
    az acr login --name $ACR_NAME
}

# Function to build and push Docker image
build_and_push_image() {
    echo -e "${YELLOW}Building and pushing Docker image...${NC}"
    
    # Build the image
    docker build -f Dockerfile.azure -t $ACR_NAME.azurecr.io/asmeranda-ml-app:latest .
    
    # Tag the image
    docker tag asmeranda-ml-app:latest $ACR_NAME.azurecr.io/asmeranda-ml-app:latest
    
    # Push to ACR
    docker push $ACR_NAME.azurecr.io/asmeranda-ml-app:latest
    
    echo -e "${GREEN}Docker image pushed successfully!${NC}"
}

# Function to create Key Vault
create_key_vault() {
    echo -e "${YELLOW}Creating Key Vault: $KEY_VAULT_NAME...${NC}"
    az keyvault create --resource-group $RESOURCE_GROUP --name $KEY_VAULT_NAME --location $LOCATION
    
    # Store secrets
    az keyvault secret set --vault-name $KEY_VAULT_NAME --name "super-admin-password" --value "Admin@12345"
    az keyvault secret set --vault-name $KEY_VAULT_NAME --name "smtp-password" --value "your-smtp-password"
}

# Function to create storage account
create_storage_account() {
    echo -e "${YELLOW}Creating Storage Account: $STORAGE_ACCOUNT...${NC}"
    az storage account create --resource-group $RESOURCE_GROUP --name $STORAGE_ACCOUNT --location $LOCATION --sku Standard_LRS
}

# Function to create Container Apps environment
create_container_apps_env() {
    echo -e "${YELLOW}Creating Container Apps Environment...${NC}"
    az containerapp env create \
        --name $CONTAINER_ENV_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION
}

# Function to deploy Container App
deploy_container_app() {
    echo -e "${YELLOW}Deploying Container App...${NC}"
    
    # Get ACR credentials
    ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
    ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
    
    # Deploy using ARM template or direct CLI
    az deployment group create \
        --resource-group $RESOURCE_GROUP \
        --template-file azure-infrastructure.json \
        --parameters \
            appName=$CONTAINER_APP_NAME \
            environmentName=$CONTAINER_ENV_NAME \
            acrName=$ACR_NAME \
            keyVaultName=$KEY_VAULT_NAME \
            storageAccountName=$STORAGE_ACCOUNT \
            location=$LOCATION
}

# Function to get application URL
get_app_url() {
    echo -e "${YELLOW}Getting application URL...${NC}"
    APP_URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
    echo -e "${GREEN}Application URL: https://$APP_URL${NC}"
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"
    
    # Create Application Insights
    az monitor app-insights component create \
        --app asmeranda-app-insights \
        --location $LOCATION \
        --resource-group $RESOURCE_GROUP
}

# Main deployment function
main() {
    check_azure_cli
    azure_login
    create_resource_group
    create_acr
    build_and_push_image
    create_key_vault
    create_storage_account
    create_container_apps_env
    deploy_container_app
    setup_monitoring
    get_app_url
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${GREEN}Your application is now running on Azure Container Apps${NC}"
    echo -e "${YELLOW}Remember to configure your SMTP settings in the application${NC}"
}

# Run main function
main "$@"