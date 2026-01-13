"""
Batch Processing Module
Process multiple diagnosis requests in parallel for high throughput.
"""

import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import time


class BatchProcessor:
    """Process multiple diagnoses in parallel for batch operations"""
    
    def __init__(self, agent, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            agent: DiagnosticAgent instance
            max_workers: Maximum parallel workers
        """
        self.agent = agent
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """
        Process batch of diagnosis requests in parallel.
        
        Args:
            requests: List of diagnosis request dicts
            
        Returns:
            List of diagnosis results
        """
        start_time = time.time()
        logger.info(f"Processing batch of {len(requests)} requests...")
        
        loop = asyncio.get_event_loop()
        
        # Create tasks for parallel execution
        tasks = [
            loop.run_in_executor(
                self.executor, 
                self._process_single, 
                req,
                i
            )
            for i, req in enumerate(requests)
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                processed_results.append({
                    "error": str(result),
                    "request_index": i
                })
            else:
                result["request_index"] = i
                processed_results.append(result)
        
        elapsed = time.time() - start_time
        logger.info(f"Batch complete: {len(requests)} requests in {elapsed:.2f}s")
        logger.info(f"Avg per request: {elapsed/len(requests)*1000:.1f}ms")
        
        return processed_results
    
    def _process_single(self, request: Dict, index: int) -> Dict:
        """Process single request (runs in thread pool)"""
        try:
            result = self.agent.diagnose_sync(
                symptoms=request.get("symptoms", ""),
                patient_id=request.get("patient_id"),
                image_path=request.get("image_path")
            )
            return result
        except Exception as e:
            logger.error(f"Error processing request {index}: {e}")
            raise
    
    def process_batch_sync(self, requests: List[Dict]) -> List[Dict]:
        """
        Synchronous wrapper for batch processing.
        
        Args:
            requests: List of diagnosis request dicts
            
        Returns:
            List of diagnosis results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_batch(requests))
        finally:
            loop.close()
    
    def __del__(self):
        """Clean up executor"""
        self.executor.shutdown(wait=False)


class BatchQueue:
    """Queue-based batch processor for continuous processing"""
    
    def __init__(self, agent, batch_size: int = 10, max_wait_seconds: float = 5.0):
        """
        Initialize batch queue.
        
        Args:
            agent: DiagnosticAgent instance
            batch_size: Maximum batch size
            max_wait_seconds: Maximum wait time before processing partial batch
        """
        self.agent = agent
        self.batch_size = batch_size
        self.max_wait = max_wait_seconds
        self.queue: asyncio.Queue = None
        self.processor = BatchProcessor(agent)
        self._running = False
        self._results: Dict[str, asyncio.Future] = {}
    
    async def start(self):
        """Start the batch queue processor"""
        self.queue = asyncio.Queue()
        self._running = True
        asyncio.create_task(self._process_loop())
        logger.info("Batch queue started")
    
    async def stop(self):
        """Stop the batch queue processor"""
        self._running = False
        logger.info("Batch queue stopped")
    
    async def submit(self, request: Dict) -> Dict:
        """
        Submit a request to the batch queue.
        
        Args:
            request: Diagnosis request
            
        Returns:
            Diagnosis result (when complete)
        """
        import uuid
        
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self._results[request_id] = future
        
        await self.queue.put({
            "request_id": request_id,
            "request": request
        })
        
        # Wait for result
        result = await future
        del self._results[request_id]
        return result
    
    async def _process_loop(self):
        """Main processing loop"""
        while self._running:
            batch = []
            start_time = time.time()
            
            # Collect batch
            while len(batch) < self.batch_size:
                try:
                    remaining_time = self.max_wait - (time.time() - start_time)
                    if remaining_time <= 0:
                        break
                    
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining_time
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                continue
            
            # Process batch
            requests = [item["request"] for item in batch]
            results = await self.processor.process_batch(requests)
            
            # Distribute results
            for i, item in enumerate(batch):
                request_id = item["request_id"]
                if request_id in self._results:
                    self._results[request_id].set_result(results[i])
