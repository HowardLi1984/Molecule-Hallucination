import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore", category=UserWarning)

tokenizer = T5Tokenizer.from_pretrained("../checkpoints/molt5_ckpts/", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('../checkpoints/molt5_ckpts/')

input_text = 'C1=CC2=C(C(=C1)[O-])NC(=CC2=O)C(=O)O'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, num_beams=5, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))