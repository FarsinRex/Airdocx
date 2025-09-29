def chunk_text(text, max_chunk_size=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        start = end
    return chunks
