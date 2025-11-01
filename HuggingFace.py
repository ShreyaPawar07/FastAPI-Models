from transformers import pipeline , AutoTokenizer
# classifier = pipeline('sentiment-analysis')

# result = classifier("I hate coding why did i didn't coded earlier!")
# print("Classification result : " ,result)

# ner = pipeline('ner')
# ner_result = ner("I m sherii, I procrastinate a lot")
# print(ner_result)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = ['Transformer are great way to person different nlp related work']

tokens = tokenizer(text,padding=True , truncation=True , return_tensors='pt')
# cls toekn is 101 and sep token is 102
print(tokens)

text2 = 'How do they tokenize things properly?'
tokens = tokenizer.tokenize(text2)
print("Tokens:",tokens)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Input IDs:",input_ids)