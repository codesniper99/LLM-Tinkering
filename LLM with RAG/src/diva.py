import transformers
from transformers.models.auto.modeling_auto import AutoModel

model = AutoModel.from_pretrained("WillHeld/DiVA-llama-3.2-1b", trust_remote_code=True)

