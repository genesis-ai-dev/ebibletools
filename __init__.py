"""
eBible Tools - Biblical text processing, translation benchmarking, and semantic search
"""

__version__ = "1.0.0"
__author__ = "Daniel Losey"

# Import main modules for easier access
from . import query
from . import metrics
from . import benchmarks

__all__ = ['query', 'metrics', 'benchmarks'] 