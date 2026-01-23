#!/bin/bash

# ASMERA NDA Cloud Deployment Scripts
# This script provides deployment commands for various cloud platforms

set -e

# Configuration
APP_NAME="asmeranda-ml-app"
VERSION="1.0.0"
REGISTRY_NAME="asmeranda-registry"

echo "🚀 ASMERA NDA Cloud Deployment Scripts"
echo "======================================"

# Function to build Docker image
build_image() {
    echo "📦 Building Docker image..."
    docker build -t ${APP_NAME}:${VERSION} .
    docker tag ${APP_NAME}:${VERSION} ${APP_NAME}:latest
    echo "✅ Docker image built successfully"
}

# Function to run locally with Docker Compose
run_local() {
    echo "🏠 Running locally with Docker Compose..."
    docker-compose up --build
}

# Function to run in detached mode
run_detached() {
    echo "🏃 Running in detached mode..."
    docker-compose up --build -d
    echo "🌐 Application available at: http://localhost:8501"
}

# Function to stop local containers
stop_local() {
    echo "🛑 Stopping local containers..."
    docker-compose down
    echo "✅ Containers stopped"
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up..."
    docker-compose down -v
    docker system prune -f
    echo "✅ Cleanup completed"
}

# AWS ECS Deployment
deploy_aws_ecs() {
    echo "☁️  Deploying to AWS ECS..."
    
    # Prerequisites: AWS CLI installed and configured
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Please install AWS CLI first."
        exit 1
    fi
    
    # Create ECR repository
    echo "📋 Creating ECR repository..."
    aws ecr create-repository --repository-name ${APP_NAME} --region us-west-2 || true
    
    # Get ECR login token
    echo "🔐 Logging into ECR..."
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com
    
    # Build and push image
    echo "📤 Pushing image to ECR..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGISTRY_URI="${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"
    
    docker build -t ${APP_NAME}:${VERSION} .
    docker tag ${APP_NAME}:${VERSION} ${REGISTRY_URI}/${APP_NAME}:${VERSION}
    docker push ${REGISTRY_URI}/${APP_NAME}:${VERSION}
    
    echo "✅ Image pushed to ECR: ${REGISTRY_URI}/${APP_NAME}:${VERSION}"
    echo "📝 Next steps:"
    echo "   1. Create ECS task definition with this image"
    echo "   2. Create ECS service"
    echo "   3. Configure Application Load Balancer"
    echo "   4. Set up EFS for persistent storage"
}

# Google Cloud Run Deployment
deploy_gcp_cloudrun() {
    echo "☁️  Deploying to Google Cloud Run..."
    
    # Prerequisites: Google Cloud CLI installed and configured
    if ! command -v gcloud &> /dev/null; then
        echo "❌ Google Cloud CLI not found. Please install gcloud first."
        exit 1
    fi
    
    # Set project
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        echo "❌ No Google Cloud project configured. Run: gcloud config set project PROJECT_ID"
        exit 1
    fi
    
    echo "📋 Using project: ${PROJECT_ID}"
    
    # Enable required APIs
    echo "🔧 Enabling required APIs..."
    gcloud services enable run.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    
    # Create Artifact Registry repository
    echo "📦 Creating Artifact Registry repository..."
    gcloud artifacts repositories create ${APP_NAME} \
        --repository-format=docker \
        --location=us-central1 \
        --description="ASMERA NDA Docker images" || true
    
    # Configure Docker authentication
    echo "🔐 Configuring Docker authentication..."
    gcloud auth configure-docker us-central1-docker.pkg.dev
    
    # Build and push image
    echo "📤 Building and pushing image..."
    IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/${APP_NAME}/${APP_NAME}:${VERSION}"
    
    docker build -t ${IMAGE_URI} .
    docker push ${IMAGE_URI}
    
    # Deploy to Cloud Run
    echo "🚀 Deploying to Cloud Run..."
    gcloud run deploy ${APP_NAME} \
        --image=${IMAGE_URI} \
        --region=us-central1 \
        --platform=managed \
        --allow-unauthenticated \
        --port=8501 \
        --memory=4Gi \
        --cpu=2 \
        --timeout=300 \
        --set-env-vars=STREAMLIT_SERVER_PORT=8501,STREAMLIT_SERVER_ADDRESS=0.0.0.0
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${APP_NAME} \
        --region=us-central1 \
        --format="value(status.url)")
    
    echo "✅ Deployed successfully!"
    echo "🌐 Application available at: ${SERVICE_URL}"
    
    # Set up Cloud Storage for persistence
    echo "💾 Setting up Cloud Storage..."
    gsutil mb -l us-central1 gs://${PROJECT_ID}-asmeranda-storage || true
    echo "📝 Storage bucket created: gs://${PROJECT_ID}-asmeranda-storage"
}

# Azure Container Instances Deployment
deploy_azure_aci() {
    echo "☁️  Deploying to Azure Container Instances..."
    
    # Prerequisites: Azure CLI installed and configured
    if ! command -v az &> /dev/null; then
        echo "❌ Azure CLI not found. Please install Azure CLI first."
        exit 1
    fi
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        echo "❌ Not logged into Azure. Run: az login"
        exit 1
    fi
    
    # Set variables
    RESOURCE_GROUP="${APP_NAME}-rg"
    LOCATION="eastus"
    ACI_NAME="${APP_NAME}-aci"
    
    echo "📋 Using resource group: ${RESOURCE_GROUP}"
    
    # Create resource group
    echo "📦 Creating resource group..."
    az group create --name ${RESOURCE_GROUP} --location ${LOCATION}
    
    # Create Azure Container Registry
    echo "📦 Creating Container Registry..."
    az acr create --resource-group ${RESOURCE_GROUP} --name ${REGISTRY_NAME} --sku Basic
    
    # Build and push image
    echo "📤 Building and pushing image..."
    ACR_LOGIN_SERVER=$(az acr show --resource-group ${RESOURCE_GROUP} --name ${REGISTRY_NAME} --query loginServer --output tsv)
    
    az acr build --registry ${REGISTRY_NAME} --image ${APP_NAME}:${VERSION} .
    
    # Deploy to Container Instances
    echo "🚀 Deploying to Container Instances..."
    az container create \
        --resource-group ${RESOURCE_GROUP} \
        --name ${ACI_NAME} \
        --image ${ACR_LOGIN_SERVER}/${APP_NAME}:${VERSION} \
        --cpu 2 \
        --memory 4 \
        --ports 8501 \
        --dns-name-label ${APP_NAME} \
        --environment-variables STREAMLIT_SERVER_PORT=8501 STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        --azure-file-volume-account-name ${APP_NAME}storage \
        --azure-file-volume-share-name ${APP_NAME}-share \
        --azure-file-volume-mount-path /app/data
    
    # Get FQDN
    FQDN=$(az container show --resource-group ${RESOURCE_GROUP} --name ${ACI_NAME} --query ipAddress.fqdn --output tsv)
    
    echo "✅ Deployed successfully!"
    echo "🌐 Application available at: http://${FQDN}:8501"
}

# DigitalOcean App Platform Deployment
deploy_digitalocean() {
    echo "☁️  Deploying to DigitalOcean App Platform..."
    
    # Prerequisites: doctl installed and authenticated
    if ! command -v doctl &> /dev/null; then
        echo "❌ doctl not found. Please install DigitalOcean CLI first."
        exit 1
    fi
    
    # Create app spec
    cat > app.yaml << EOF
name: ${APP_NAME}
services:
- name: web
  source_dir: /
  github:
    repo: your-username/asmeranda
    branch: main
  run_command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
  environment_slug: python
  instance_count: 1
  instance_size_slug: professional-xs
  env:
  - key: STREAMLIT_SERVER_PORT
    value: "8501"
  - key: STREAMLIT_SERVER_ADDRESS
    value: "0.0.0.0"
  http_port: 8501
  routes:
  - path: /
EOF
    
    echo "📋 Created app.yaml specification"
    echo "📝 Next steps:"
    echo "   1. Push your code to GitHub"
    echo "   2. Run: doctl apps create --spec app.yaml"
    echo "   3. Configure environment variables and storage"
}

# Health check function
health_check() {
    echo "🔍 Performing health check..."
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        echo "✅ Application is healthy"
    else
        echo "❌ Application is not responding"
        exit 1
    fi
}

# Show logs
show_logs() {
    echo "📋 Showing application logs..."
    docker-compose logs -f asmeranda-app
}

# Main menu
case "$1" in
    "build")
        build_image
        ;;
    "run")
        run_local
        ;;
    "start")
        run_detached
        ;;
    "stop")
        stop_local
        ;;
    "clean")
        cleanup
        ;;
    "health")
        health_check
        ;;
    "logs")
        show_logs
        ;;
    "deploy-aws")
        build_image
        deploy_aws_ecs
        ;;
    "deploy-gcp")
        build_image
        deploy_gcp_cloudrun
        ;;
    "deploy-azure")
        build_image
        deploy_azure_aci
        ;;
    "deploy-digitalocean")
        deploy_digitalocean
        ;;
    *)
        echo "Usage: $0 {build|run|start|stop|clean|health|logs|deploy-aws|deploy-gcp|deploy-azure|deploy-digitalocean}"
        echo ""
        echo "Local Development:"
        echo "  build          - Build Docker image"
        echo "  run            - Run with Docker Compose (interactive)"
        echo "  start          - Run in detached mode"
        echo "  stop           - Stop containers"
        echo "  clean          - Clean up containers and images"
        echo "  health         - Check application health"
        echo "  logs           - Show application logs"
        echo ""
        echo "Cloud Deployment:"
        echo "  deploy-aws     - Deploy to AWS ECS"
        echo "  deploy-gcp     - Deploy to Google Cloud Run"
        echo "  deploy-azure   - Deploy to Azure Container Instances"
        echo "  deploy-digitalocean - Deploy to DigitalOcean App Platform"
        exit 1
        ;;
esac
