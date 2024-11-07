import torch
from sentence_transformers import SentenceTransformer

class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    @torch.no_grad()
    def __call__(self, input):
        x = self.model.encode(input, show_progress_bar=False,
                              convert_to_tensor=True)
        return x.cpu()

if __name__ == 'main':
    encoder = SequenceEncoder()