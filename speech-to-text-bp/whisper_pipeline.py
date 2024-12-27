import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.cpu
else:
    print("Using GPU")

class WhisperPipeline:
    def __init__(self, model_name: str, chunk_length_s: int, device: str, generate_kwargs: dict):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=chunk_length_s,
            device=device,
            generate_kwargs=generate_kwargs
        )
        
    def __call__(self, data_sample: str, batch_size: int):
        return self.pipe(data_sample, batch_size=batch_size)
