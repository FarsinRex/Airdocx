# Example in-memory store (replace with real embeddings if needed)
class EmbedStore:
    def __init__(self):
        self.data = []

    def add_texts(self, batch):
        # batch = list of {"text":..., "meta":...}
        self.data.extend(batch)
