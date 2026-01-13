"""
A/B Testing Framework
Route requests between model versions and track experiment metrics.
"""

import hashlib
import random
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger


class ABTestRouter:
    """Route requests between model versions for A/B testing"""
    
    def __init__(self, experiments: Dict = None):
        """
        Initialize A/B test router.
        
        Args:
            experiments: Dict of experiment configs
        """
        self.experiments = experiments or {}
        self.results = {}  # Track results per experiment
    
    def add_experiment(
        self,
        name: str,
        variants: Dict[str, int],
        metrics: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        """
        Add a new A/B experiment.
        
        Args:
            name: Experiment identifier
            variants: Dict of variant names to traffic percentages
            metrics: List of metrics to track
            start_date: When experiment starts
            end_date: When experiment ends
        """
        if sum(variants.values()) != 100:
            raise ValueError("Variant percentages must sum to 100")
        
        self.experiments[name] = {
            "variants": variants,
            "metrics": metrics or ["accuracy", "latency"],
            "start_date": start_date or datetime.utcnow(),
            "end_date": end_date,
            "active": True
        }
        
        self.results[name] = {variant: [] for variant in variants}
        
        logger.info(f"Added experiment '{name}' with variants: {list(variants.keys())}")
    
    def get_variant(self, user_id: str, experiment_name: str) -> str:
        """
        Determine which variant to use for a user.
        Uses consistent hashing for deterministic assignment.
        
        Args:
            user_id: Unique user/request identifier
            experiment_name: Name of the experiment
            
        Returns:
            Variant name
        """
        experiment = self.experiments.get(experiment_name)
        
        if not experiment or not experiment["active"]:
            return "control"
        
        # Check date validity
        now = datetime.utcnow()
        if experiment.get("end_date") and now > experiment["end_date"]:
            logger.info(f"Experiment '{experiment_name}' has ended")
            return "control"
        
        # Consistent hashing for user assignment
        hash_input = f"{user_id}:{experiment_name}"
        user_hash = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100
        
        cumulative = 0
        for variant, percentage in experiment["variants"].items():
            cumulative += percentage
            if user_hash < cumulative:
                return variant
        
        return "control"
    
    def record_result(
        self,
        experiment_name: str,
        variant: str,
        metrics: Dict
    ):
        """
        Record result for an experiment variant.
        
        Args:
            experiment_name: Name of the experiment
            variant: Variant that was used
            metrics: Dict of metric name -> value
        """
        if experiment_name not in self.results:
            self.results[experiment_name] = {}
        
        if variant not in self.results[experiment_name]:
            self.results[experiment_name][variant] = []
        
        self.results[experiment_name][variant].append({
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        })
    
    def get_experiment_stats(self, experiment_name: str) -> Dict:
        """
        Get aggregated statistics for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dict with statistics per variant
        """
        if experiment_name not in self.results:
            return {"error": "Experiment not found"}
        
        stats = {}
        
        for variant, results in self.results[experiment_name].items():
            if not results:
                stats[variant] = {"sample_size": 0}
                continue
            
            # Aggregate metrics
            sample_size = len(results)
            metric_sums = {}
            
            for result in results:
                for metric, value in result["metrics"].items():
                    if metric not in metric_sums:
                        metric_sums[metric] = []
                    metric_sums[metric].append(value)
            
            # Calculate means
            metric_means = {
                metric: sum(values) / len(values)
                for metric, values in metric_sums.items()
            }
            
            stats[variant] = {
                "sample_size": sample_size,
                "metrics": metric_means
            }
        
        # Calculate winner
        if len(stats) >= 2:
            experiment = self.experiments.get(experiment_name, {})
            primary_metric = experiment.get("metrics", ["accuracy"])[0]
            
            best_variant = None
            best_value = -float('inf')
            
            for variant, data in stats.items():
                if "metrics" in data and primary_metric in data["metrics"]:
                    value = data["metrics"][primary_metric]
                    if value > best_value:
                        best_value = value
                        best_variant = variant
            
            stats["_winner"] = best_variant
            stats["_primary_metric"] = primary_metric
        
        return stats
    
    def end_experiment(self, experiment_name: str) -> Dict:
        """
        End an experiment and return final results.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Final statistics
        """
        if experiment_name in self.experiments:
            self.experiments[experiment_name]["active"] = False
            self.experiments[experiment_name]["end_date"] = datetime.utcnow()
        
        return self.get_experiment_stats(experiment_name)


# Singleton router for global access
_ab_router = None

def get_ab_router() -> ABTestRouter:
    """Get global A/B test router"""
    global _ab_router
    if _ab_router is None:
        _ab_router = ABTestRouter()
        
        # Add default experiments
        _ab_router.add_experiment(
            name="new_nlp_model",
            variants={"control": 90, "treatment": 10},
            metrics=["accuracy", "latency", "confidence"]
        )
    
    return _ab_router


# Example usage
if __name__ == "__main__":
    router = get_ab_router()
    
    # Simulate requests
    for i in range(100):
        user_id = f"user_{i}"
        variant = router.get_variant(user_id, "new_nlp_model")
        
        # Simulate metrics
        accuracy = 0.85 + (0.05 if variant == "treatment" else 0) + random.uniform(-0.1, 0.1)
        latency = 200 + ((-20) if variant == "treatment" else 0) + random.uniform(-50, 50)
        
        router.record_result("new_nlp_model", variant, {
            "accuracy": accuracy,
            "latency": latency
        })
    
    # Get stats
    stats = router.get_experiment_stats("new_nlp_model")
    print("Experiment Results:")
    for variant, data in stats.items():
        if not variant.startswith("_"):
            print(f"  {variant}: {data}")
    
    print(f"  Winner: {stats.get('_winner')}")
