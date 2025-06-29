# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
model = AutoModelForSequenceClassification.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
