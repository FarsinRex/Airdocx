import os
from dotenv import load_dotenv
import pinecone 

load_dotenv()
pinecone.init(api_key = os.getenv("PINECONE_API_KEY"), environment="us-east-1")

index_name = os.getenv("PINECONE_INDEX_NAME")
available_indexes = pinecone.Index(index_name)
stats = available_indexes.describe_index_stats()
print(stats)
