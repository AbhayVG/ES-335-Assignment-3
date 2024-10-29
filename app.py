import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Streamlit app on Text Generation
st.title('Text Generation')

# select random seed (drop-down)
seed = st.selectbox('Select the random seed', (1, 2 ,3))

np.random.seed(seed)
torch.manual_seed(seed)

# select embedding dimension (drop-down)
emb_dim = st.selectbox('Select the embedding dimension', (32, 64))

# select context length (drop-down)
block_size = st.selectbox('Select the context length', (2, 6 ,10))

activation_functions = {"ReLU": 1, "Tanh": 2, "Sigmoid": 3, "Sin": 4}

# select activation function (drop-down)
activation = st.selectbox('Select the activation function', ('ReLU', 'Tanh', 'Sigmoid', 'Sin'))

# path to the model
path = f"models/{seed}_context_{block_size}_embedding_{emb_dim}_activation_{activation_functions[activation]}.pt"

# select "k" (number of characters to predict)
k = st.slider('Select the number of characters to predict', 1, 100, 1)

# Enter the text to predict upon
text = st.text_input('Enter the text to predict upon', 'Type here...')

# read data
with open(r'D:\Users\abhay\Downloads\ML_Classes\ES335-Assignment-3\dataset.txt', 'r', encoding='utf-8') as file:
    data = file.read()
# unique characters
unique_chars = list(set(''.join(data)))
unique_chars.sort()

# create vocabulary
vocab_dict = {i:ch for i, ch in enumerate(unique_chars)}
vocab_dict_inv = {ch:i for i, ch in enumerate(unique_chars)}

# model
class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, 1024)
    self.lin2 = nn.Linear(1024, vocab_size)
    self.activation = nn.ReLU()

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = self.activation(self.lin1(x))
    x = self.lin2(x)
    return x

# create model
model = NextChar(block_size, len(unique_chars), emb_dim)

# load model weights and move to CPU
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()

# function to generate next character
def next_char(model, sentence, itos, stoi, block_size, max_len=10):
    if len(sentence) < block_size:
        sentence = " " * (block_size - len(sentence)) + sentence
    using_for_predicction = sentence[-block_size:].lower()
    context = [stoi[word] for word in using_for_predicction]
    prediction = ""
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        prediction += ch
        context = context[1:] + [ix]

    return prediction

# generate next k characters
generation = next_char(model, text, vocab_dict, vocab_dict_inv, block_size, k)

# if there is a space in the prediction, replace it with | | for better visualization
generation = generation.replace(' ', '| |')

# if there is a newline in the prediction, replace it with \n for better visualization
# generation = generation.replace('\n', '|\n|')

# write prediction
st.write(f'Predicted next :red[{k}] character{" is" if k==1 else "s are"}:')
st.success(generation)

# note
st.info('Note: "| |" represents a space.')

# feature visualization (t-SNE)

# with random state set to 42, the plot will be reproducible
emb = model.emb

# Assuming you have a tensor of higher-dimensional embeddings
embeddings = (emb.weight.data).detach().numpy()

# Apply t-SNE
tsne = TSNE(n_components=min(2, emb_dim), perplexity=30, n_iter=300)
embeddings_tsne = tsne.fit_transform(embeddings)

# function to plot t-SNE of the learnt embeddings
def plot_emb(embeddings_tsne, itos, dim, ax=None):
	if ax is None:
		_, ax = plt.subplots()

	for i in range(len(itos)):
		char = itos[i]
		if char == ' ':
			char = '| |'
		elif char == '\n':
			char = '\\n'
		x = embeddings_tsne[i, 0]
		y = embeddings_tsne[i, 1]
		ax.scatter(x, y, color='k')
		ax.text(x + 0.07, y + 0.07, char)
	return ax

# convert the embeddings_tsne to a dataframe
df = pd.DataFrame(embeddings_tsne, columns=['1', '2'])

# plot the embeddings
fig, ax = plt.subplots()
ax = plot_emb(embeddings_tsne, vocab_dict, emb_dim, ax=ax)
ax.set_title('t-SNE of the learnt Embeddings')
st.pyplot(fig)

# note
st.info('Note: "| |" represents a space and "\\n" represents a newline.')