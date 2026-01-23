# 🚀 ASMERA NDA Cloud Deployment Quickstart

## 📋 Prerequisites

### Required Software
- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **Git**

### Cloud Provider CLI (choose one)
- **AWS**: AWS CLI v2
- **Google Cloud**: Google Cloud CLI
- **Azure**: Azure CLI
- **DigitalOcean**: doctl

## 🏃‍♂️ Quick Start (5 Minutes)

### 1. Local Testing
```bash
# Clone repository (if not already)
git clone <your-repo-url>
cd Asmeranda

# Make deployment script executable
chmod +x cloud_deployment_scripts.sh

# Build and run locally
./cloud_deployment_scripts.sh build
./cloud_deployment_scripts.sh start

# Access at: http://localhost:8501
```

### 2. Choose Cloud Provider & Deploy

#### 🟢 Google Cloud Run (Recommended - Most Cost-Effective)
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Configure
gcloud init
gcloud auth login

# Deploy
./cloud_deployment_scripts.sh deploy-gcp
```

**Cost**: ~$63-128/month | **Time**: 5 minutes | **Maintenance**: Minimal

#### 🟠 AWS ECS (Enterprise-Grade)
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure
aws configure

# Deploy
./cloud_deployment_scripts.sh deploy-aws
```

**Cost**: ~$95-175/month | **Time**: 10 minutes | **Maintenance**: Moderate

#### 🔵 Azure Container Instances
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Configure
az login

# Deploy
./cloud_deployment_scripts.sh deploy-azure
```

**Cost**: ~$80-148/month | **Time**: 8 minutes | **Maintenance**: Moderate

## 📊 Deployment Comparison

| Provider | Cost/Month | Setup Time | Scaling | Storage | Best For |
|----------|------------|------------|---------|---------|----------|
| **Google Cloud Run** | $63-128 | 5 min | ⭐⭐⭐⭐⭐ | Cloud Storage | Startups, SMB |
| **AWS ECS** | $95-175 | 10 min | ⭐⭐⭐⭐ | EFS | Enterprise |
| **Azure ACI** | $80-148 | 8 min | ⭐⭐⭐ | Azure Files | Azure users |
| **DigitalOcean** | $70-120 | 15 min | ⭐⭐⭐ | Spaces | Simplicity |

## 🔧 Configuration Options

### Environment Variables
```bash
# Performance tuning
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
MAX_UPLOAD_SIZE=100MB
CACHE_TTL=86400

# Resource allocation
MEMORY_LIMIT=4Gi
CPU_LIMIT=2
TIMEOUT=300
```

### Storage Configuration
```bash
# Required storage paths
/models/           # Trained models (1-5 GB)
/uploads/          # User uploads (100 MB - 1 GB)
/interpretation_cache/ # SHAP/LIME cache (500 MB - 2 GB)
```

## 🛠️ Management Commands

### Local Development
```bash
# Build image
./cloud_deployment_scripts.sh build

# Run interactively
./cloud_deployment_scripts.sh run

# Run in background
./cloud_deployment_scripts.sh start

# View logs
./cloud_deployment_scripts.sh logs

# Stop application
./cloud_deployment_scripts.sh stop

# Clean up everything
./cloud_deployment_scripts.sh clean

# Health check
./cloud_deployment_scripts.sh health
```

### Cloud Management
```bash
# Update deployment
./cloud_deployment_scripts.sh deploy-[provider]

# Monitor logs (cloud-specific)
gcloud logs read "resource.type=cloud_run_revision"  # GCP
aws logs tail /aws/ecs/asmeranda-ml-app             # AWS
az container logs show --follow                     # Azure
```

## 📈 Performance Optimization

### Recommended Settings
```yaml
# Cloud Run (GCP)
resources:
  cpu: 2
  memory: 4Gi
  concurrency: 10
  max_instances: 100

# ECS (AWS)
cpu: 1024
memory: 4096
min_capacity: 1
max_capacity: 10

# ACI (Azure)
cpu: 2
memory: 4
os_type: Linux
restart_policy: Always
```

### Auto-Scaling Configuration
```bash
# GCP Cloud Run
gcloud run services update asmeranda-ml-app \
  --max-instances=100 \
  --cpu-throttling=false

# AWS ECS
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/asmeranda-cluster/asmeranda-ml-app \
  --min-capacity 1 \
  --max-capacity 10
```

## 🔒 Security Setup

### SSL/TLS (Automatic)
- **GCP**: Built-in HTTPS
- **AWS**: ACM certificate with ALB
- **Azure**: Built-in HTTPS
- **DigitalOcean**: Built-in HTTPS

### Access Control
```bash
# GCP IAM
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:email@example.com" \
  --role="roles/run.invoker"

# AWS IAM
aws iam create-policy \
  --policy-name AsmerandaAccessPolicy \
  --policy-document file://policy.json

# Azure RBAC
az role assignment create \
  --assignee user@domain.com \
  --role Contributor \
  --resource-group asmeranda-rg
```

## 💰 Cost Optimization

### Money-Saving Tips
1. **Use serverless** (Cloud Run) for variable traffic
2. **Set minimum instances** to 0 for cost savings
3. **Use smaller instances** for development
4. **Enable auto-scaling** to match demand
5. **Monitor usage** and adjust resources

### Cost Monitoring
```bash
# GCP
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Asmeranda Budget" \
  --budget-amount=200USD

# AWS
aws budgets create-budget \
  --account-id ACCOUNT_ID \
  --budget-name AsmerandaBudget \
  --budget-type COST \
  --time-unit MONTHLY \
  --budget-amount 200

# Azure
az consumption budget create \
  --name AsmerandaBudget \
  --category cost \
  --amount 200 \
  --time-grain Monthly
```

## 🔍 Monitoring & Logging

### Health Checks
```bash
# Application health
curl http://localhost:8501/_stcore/health

# Container health
docker ps
docker stats

# Cloud health
gcloud run services describe asmeranda-ml-app
aws ecs describe-services --services asmeranda-ml-app
```

### Log Collection
```bash
# Stream logs in real-time
./cloud_deployment_scripts.sh logs

# Cloud-specific logs
gcloud logs tail "resource.type=cloud_run_revision"
aws logs tail /aws/ecs/asmeranda-ml-app
az container logs show --follow
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker logs asmeranda-ml-app

# Common fixes:
# - Check requirements.txt
# - Verify Dockerfile syntax
# - Check port conflicts
```

#### 2. Memory Issues
```bash
# Increase memory allocation
# GCP: --memory=8Gi
# AWS: memory: 8192
# Azure: --memory 8
```

#### 3. Storage Issues
```bash
# Check storage mounts
docker exec -it asmeranda-ml-app ls -la /app/models

# Fix permissions
docker exec asmeranda-ml-app chown -R appuser:appuser /app
```

#### 4. Network Issues
```bash
# Check connectivity
curl -I http://localhost:8501

# Check firewall
telnet localhost 8501
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Optimize by:
# - Increasing CPU/memory
# - Enabling caching
# - Using CDN for static assets
```

## 📚 Additional Resources

### Documentation
- [Docker Documentation](https://docs.docker.com/)
- [Google Cloud Run Guide](https://cloud.google.com/run/docs)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Azure Container Instances](https://docs.microsoft.com/azure/container-instances/)

### Support
- **Application Issues**: Check logs first
- **Cloud Issues**: Use cloud provider support
- **Docker Issues**: Docker documentation

### Community
- [Streamlit Community](https://discuss.streamlit.io/)
- [Docker Hub](https://hub.docker.com/)
- [Cloud Provider Forums]

---

## 🎯 Recommended Deployment Path

### For Production (Recommended)
1. **Start with Google Cloud Run** (5 minutes, $63/month)
2. **Monitor usage** for 1 week
3. **Optimize resources** based on actual usage
4. **Set up alerts** for cost and performance

### For Development
1. **Use Docker Compose** locally
2. **Deploy to staging** on Cloud Run
3. **Test thoroughly** before production
4. **Promote to production** when ready

### For Enterprise
1. **Use AWS ECS** with EFS
2. **Set up VPC** isolation
3. **Configure monitoring** and alerting
4. **Implement backup** and disaster recovery

---

**🎉 Congratulations!** Your ASMERA NDA ML application is now ready for cloud deployment with zero code changes required.
