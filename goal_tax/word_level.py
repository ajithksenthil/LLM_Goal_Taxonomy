import gensim.downloader as api
from gensim.models import KeyedVectors

# Load the pre-trained Word2Vec model
model_path = 'GoogleNews-vectors-negative300.bin.gz'
print("Loading the Word2Vec model...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Model loaded successfully.")

# Define your taxonomy words
taxonomy = ['king', 'queen', 'apple', 'banana', 'car', 'dog']

# Set the number of similar words to retrieve
topn = 5

# For each word in the taxonomy, find similar words
for word in taxonomy:
    if word in model:
        similar_words = model.most_similar(positive=[word], topn=topn)
        print(f"\nWord: {word}")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word} (similarity: {similarity:.3f})")
    else:
        print(f"\nWord '{word}' not in vocabulary.")
