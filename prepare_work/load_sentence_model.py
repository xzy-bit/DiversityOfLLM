# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-large-nli-stsb-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-large-nli-stsb-mean-tokens")
