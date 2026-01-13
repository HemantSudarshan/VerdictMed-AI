#!/bin/bash
# CDSS Rollback Script
# Rolls back to previous deployment version

set -e

# Configuration
DEPLOYMENT_NAME="${1:-cdss-prod}"
NAMESPACE="${2:-default}"

echo "=========================================="
echo "CDSS Rollback Script"
echo "=========================================="
echo "Deployment: $DEPLOYMENT_NAME"
echo "Namespace: $NAMESPACE"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

# Get rollout history
echo "📋 Current rollout history:"
kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE

# Get current revision
CURRENT=$(kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE | tail -1 | awk '{print $1}')
PREVIOUS=$((CURRENT - 1))

if [ "$PREVIOUS" -lt 1 ]; then
    echo "❌ No previous revision to rollback to"
    exit 1
fi

echo ""
echo "🔄 Rolling back from revision $CURRENT to $PREVIOUS..."

# Perform rollback
kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE --to-revision=$PREVIOUS

# Wait for rollout
echo ""
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

# Verify health
echo ""
echo "🔍 Verifying health..."

# Get pod status
READY=$(kubectl get deployment/$DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
DESIRED=$(kubectl get deployment/$DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.spec.replicas}')

if [ "$READY" == "$DESIRED" ]; then
    echo "✅ Rollback complete! $READY/$DESIRED pods ready"
else
    echo "⚠️ Rollback complete but pods not fully ready: $READY/$DESIRED"
fi

# Health check endpoint
SERVICE_IP=$(kubectl get svc/$DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

if [ -n "$SERVICE_IP" ]; then
    echo ""
    echo "🏥 Health check:"
    curl -sf "http://$SERVICE_IP/health" && echo " ✅ Healthy" || echo " ❌ Health check failed"
fi

echo ""
echo "=========================================="
echo "Rollback Summary"
echo "=========================================="
echo "Previous revision: $CURRENT"
echo "Current revision: $PREVIOUS"
echo "Status: Complete"
