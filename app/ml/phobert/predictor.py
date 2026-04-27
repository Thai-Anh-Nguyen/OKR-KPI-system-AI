"""
PhoBERT predictor module.

This module is intentionally thin — the actual inference logic lives in
app/services/phobert_svc.py, which calls the pipeline from loader.py.
This file can be extended for batch prediction or custom post-processing.
"""
