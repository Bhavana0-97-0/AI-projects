import nltk
nltk.download('popular')
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from nltk.corpus import brown
nltk.download('brown')
brown.words()
nltk.download('punkt_tab')

warnings.filterwarnings(action='ignore')

# Read text file
sample = open(r"alice_in_wonderland.txt", encoding='utf-8')
s = sample.read()

# Replace escape characters
f = s.replace("\n", " ")

data = []

# Tokenize into sentences, then words
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

# Train CBOW model
model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=0)

# Define 10 word pairs
word_pairs = [
    ("alice", "rabbit"),
    ("queen", "king"),
    ("alice", "sister"),
    ("rabbit", "watch"),
    ("garden", "flowers"),
    ("mouse", "pool"),
    ("door", "key"),
    ("drink", "bottle"),
    ("mad", "hatter"),
    ("sleep", "dream")
]

# Calculate and print cosine similarities
similarities = []
print("Cosine Similarities (CBOW):\n")
for w1, w2 in word_pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        similarities.append(((w1, w2), sim))
        print(f"{w1:>10} and {w2:<10} : {sim:.4f}")
    else:
        similarities.append(((w1, w2), None))
        print(f"{w1:>10} and {w2:<10} : Not in vocabulary")

    # Identify most similar pair
valid_similarities = [item for item in similarities if item[1] is not None]
most_similar = max(valid_similarities, key=lambda x: x[1])
print(f"\nMost similar pair: {most_similar[0][0]} and {most_similar[0][1]} with similarity{most_similar[1]: .4f}")