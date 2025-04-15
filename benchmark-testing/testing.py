"""
Title: Whisper AI Benchmarking Script

Summary:
This script evaluates Whisper AI models on key performance metrics such as:
- Inference Speed (latency)
- VRAM Usage
- Word Error Rate (WER)

Authors: Alex Darwiche
Date: 2025-03-19  

"""

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")