from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#this puts the model in eval mode (no training)
model.eval()

#turns text into numbers
inputs = tokenizer("the cat sat on the", return_tensors='pt')

#run the numbers through the model
outputs = model(**inputs)

#prints next predicted token
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Output shape: {outputs.logits.shape}")
print('Successful, model loaded and ran')