from abc import ABC, abstractmethod
import psutil
import time
from typing import Dict, Tuple, Optional

class MLProgressAdapter(ABC):
    @abstractmethod
    def calculate_progress(self, cpu_times: Dict[int, float], total_fits: int) -> Tuple[float, float]:
        """Returns (completed_fits, estimated_hours_remaining)"""
        pass

class SVRAdapter(MLProgressAdapter):
    def calculate_progress(self, cpu_times: Dict[int, float], total_fits: int) -> Tuple[float, float]:
        time_per_fit = 1200  # SVR specific
        avg_cpu = sum(cpu_times.values()) / len(cpu_times)
        fits_per_worker = avg_cpu / time_per_fit
        total_progress = fits_per_worker * len(cpu_times)
        remaining = (total_fits - total_progress) * time_per_fit / (len(cpu_times) * 3600)
        return total_progress, remaining

class XGBoostAdapter(MLProgressAdapter):
    def calculate_progress(self, cpu_times: Dict[int, float], total_fits: int) -> Tuple[float, float]:
        time_per_fit = 300  # XGBoost specific
        # Similar calculation but with XGBoost parameters
        pass

class MLMonitor:
    def __init__(self, algorithm: str, total_fits: int):
        self.adapters = {
            'svr': SVRAdapter(),
            'xgboost': XGBoostAdapter(),
            # Add more adapters
        }
        self.adapter = self.adapters[algorithm.lower()]
        self.total_fits = total_fits
        self.tracker = WorkerTracker()
    
    def update(self, process: psutil.Process) -> Tuple[float, float]:
        cpu_times = {c.pid: c.cpu_times().user + c.cpu_times().system 
                    for c in process.children() 
                    if c.cpu_times().user > 100}
        return self.adapter.calculate_progress(cpu_times, self.total_fits)