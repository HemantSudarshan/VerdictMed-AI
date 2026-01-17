# CDSS Incident Response Playbook

Comprehensive runbook for responding to CDSS system incidents and alerts.

---

## Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| P1 | Service down, false negatives | < 15 min | On-call + manager |
| P2 | Accuracy < 85%, high latency | < 1 hour | On-call |
| P3 | Elevated escalations, drift | < 4 hours | Next business day |

---

## Alert Runbooks

### AccuracyDropped (P1/P2)

**Trigger**: Model accuracy < 85% for 5 minutes

**Immediate Actions**:
1. Check recent deployments:
   ```bash
   kubectl rollout history deployment/cdss-prod
   ```

2. Compare model versions in A/B test:
   ```bash
   curl http://prometheus:9090/api/v1/query?query=cdss_active_model_version
   ```

3. Check data distribution shift:
   - Review recent diagnosis patterns in Grafana
   - Compare symptom distributions vs. baseline

4. If caused by new deployment, rollback immediately:
   ```bash
   ./scripts/rollback.sh
   ```

5. If persistent after rollback:
   - Page ML team
   - Initiate emergency model retraining
   - Consider switching to previous stable model version

**Verification**:
- Accuracy returns to > 85%
- No increase in false negatives
- Doctor feedback remains positive

---

### HighLatency (P2)

**Trigger**: P95 latency > 5 seconds for 2 minutes

**Immediate Actions**:
1. Check pod resource usage:
   ```bash
   kubectl top pods -l app=cdss
   ```

2. Check database latency:
   - Neo4j: `CALL dbms.queryJmx('org.neo4j:*') YIELD name, attributes`
   - Redis: `redis-cli --latency`
   - Weaviate: Check /v1/meta endpoint

3. Check cache hit rate:
   ```bash
   curl http://cdss-api:8000/metrics | grep cache_hits
   ```

4. Scale up if resource-constrained:
   ```bash
   kubectl scale deployment/cdss-prod --replicas=5
   ```

5. Enable aggressive caching:
   - Increase Redis TTL temporarily
   - Enable model response caching

**Verification**:
- P95 latency < 2 seconds
- CPU/Memory utilization < 70%
- Cache hit rate > 60%

---

### FalseNegativeSpike (P1)

**Trigger**: > 5 false negatives in 1 hour

⚠️ **CRITICAL PATIENT SAFETY INCIDENT**

**Immediate Actions**:
1. **IMMEDIATELY review all missed cases**:
   ```sql
   SELECT * FROM diagnoses 
   WHERE doctor_confirmed = false 
   AND created_at > NOW() - INTERVAL '1 hour'
   ORDER BY severity DESC;
   ```

2. Identify affected disease categories:
   - Are false negatives specific to certain conditions?
   - Check if critical conditions (MI, stroke, sepsis) are affected

3. Check recent changes:
   - Model deployment: `git log --since="1 hour ago" --oneline`
   - Data ingestion changes
   - Configuration updates

4. **If patient safety at risk, take system offline**:
   ```bash
   kubectl scale deployment/cdss-prod --replicas=0
   ```

5. Emergency escalation:
   - Page ML team lead
   - Notify medical advisory board
   - Prepare incident report for regulatory review

**Post-Incident**:
- Root cause analysis within 24 hours
- Retrain model on missed cases
- Update safety validation rules
- Review incident with medical team

---

### EscalationRateHigh (P3)

**Trigger**: Escalation rate > 30% for 10 minutes

**Actions**:
1. Check confidence threshold settings:
   ```bash
   kubectl get configmap cdss-config -o yaml | grep threshold
   ```

2. Review recent cases being escalated:
   - Are symptoms vague/incomplete?
   - Is there a spike in rare conditions?
   - Check if specific modality (image/NLP) is failing

3. Analyze patterns:
   ```sql
   SELECT escalation_reason, COUNT(*) 
   FROM diagnoses 
   WHERE needs_escalation = true 
   AND created_at > NOW() - INTERVAL '1 hour'
   GROUP BY escalation_reason;
   ```

4. Adjust thresholds if appropriate (requires approval):
   - Lower min_confidence_threshold by 0.05
   - Update safety validation rules

**Verification**:
- Escalation rate returns to 15-25% baseline
- No decrease in diagnostic accuracy
- Doctor override rate remains stable

---

### ServiceDown (P1)

**Trigger**: CDSS API unavailable for 1 minute

**Immediate Actions**:
1. Check pod status:
   ```bash
   kubectl get pods -l app=cdss
   kubectl describe pod <pod-name>
   ```

2. Check service logs:
   ```bash
   kubectl logs deployment/cdss-prod --tail=100
   ```

3. Check dependencies:
   - Neo4j: `curl http://neo4j:7474`
   - Redis: `redis-cli ping`
   - Weaviate: `curl http://weaviate:8080/v1/.well-known/ready`

4. Restart deployment if needed:
   ```bash
   kubectl rollout restart deployment/cdss-prod
   ```

5. If persistent, rollback to last known good:
   ```bash
   ./scripts/rollback.sh
   ```

**Verification**:
- Health endpoint responds: `curl http://cdss-api:8000/health`
- Can complete test diagnosis
- All dependencies healthy

---

### LowDailyVolume (Info)

**Trigger**: < 100 diagnoses in 24 hours

**Actions**:
1. Check if planned downtime or maintenance

2. Verify integration with EHR systems:
   - Check API keys/credentials
   - Review network connectivity

3. Check user adoption:
   - Survey recent doctor feedback
   - Review usage dashboard

4. Marketing/training needed:
   - Schedule doctor training session
   - Send usage reminder to clinical staff

---

## Escalation Contacts

| Role | Contact | When to Escalate |
|------|---------|------------------|
| On-call Engineer | Slack: #cdss-oncall | All P1/P2 alerts |
| ML Team Lead | [Contact] | False negatives, accuracy drops |
| DevOps Lead | [Contact] | Infrastructure issues |
| Medical Director | [Contact] | Patient safety concerns |
| Regulatory Affairs | [Contact] | Compliance incidents |

---

## Post-Incident Process

1. **Immediate** (< 2 hours):
   - Document timeline in incident log
   - Notify stakeholders
   - Implement temporary fix

2. **Short-term** (< 24 hours):
   - Root cause analysis
   - Permanent fix implementation
   - Update monitoring/alerts

3. **Long-term** (< 1 week):
   - Post-mortem meeting
   - Update documentation
   - Implement preventive measures
   - Share learnings with team

---

## Monitoring Dashboard

Access key dashboards:
- **Operations**: http://grafana:3000/d/cdss-ops
- **Model Performance**: http://grafana:3000/d/cdss-ml
- **Alerts**: http://alertmanager:9093

---

## Additional Resources

- [System Architecture](../architecture.md)
- [Deployment Guide](../deployment.md)
- [Model Retraining Procedure](../ml/retraining.md)
- [Safety Validation Rules](../safety/validation.md)
