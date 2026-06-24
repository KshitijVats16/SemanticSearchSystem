class EmbeddingModel:
    def __init__(self, dim=DEFAULT_DIM, model_dir=MODEL_DIR):
        self.dim = dim
        self.model_dir = Path(model_dir)