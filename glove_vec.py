from nltk.tokenize import word_tokenize
from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim
from nltk.stem import WordNetLemmatizer


class Glove():
	def __init__(self, text, incl_stopwords = True, lemmatization = False):
		# Set parameters
		self.context_size = 6
		self.embed_size = 1024
		self.xmax = 2
		self.alpha = 0.075
		self.batch_size = 200
		self.l_rate = 0.05
		self.num_epochs = 10
		self.incl_stopwords = incl_stopwords
		self.lemmatization = lemmatization
		self.word_list = []
		# Create vocabulary and word lists
		wordnet_lemmatizer = WordNetLemmatizer()
		for line in text:
			temp_line = []
			for word in line:
				if lemmatization:
					self.word_list.append(wordnet_lemmatizer.lemmatize(word))
				else:
					self.word_list.append(word)
		self.vocab = np.unique(self.word_list)
		self.w_list_size = len(self.word_list)
		self.vocab_size = len(self.vocab)

		# check for cuda
		device = None
		if torch.cuda.is_available():
			device = torch.device(torch.cuda.current_device())
		else:
			device = torch.device("cpu")

		# Create word to index mapping
		self.w_to_i = {word: ind for ind, word in enumerate(self.vocab)}

		# Construct co-occurence matrix
		self.comat = np.zeros((self.vocab_size, self.vocab_size))
		for i in range(self.w_list_size):
			for j in range(1, self.context_size+1):
				ind = self.w_to_i[self.word_list[i]]
				if i-j > 0:
					lind = self.w_to_i[self.word_list[i-j]]
					self.comat[ind, lind] += 1.0/j
				if i+j < self.w_list_size:
					rind = self.w_to_i[self.word_list[i+j]]
					self.comat[ind, rind] += 1.0/j

		# Non-zero co-occurrences
		self.coocs = np.transpose(np.nonzero(self.comat))
		# Set up word vectors and biases
		self.l_embed, self.r_embed = [
			[Variable(torch.from_numpy(np.random.normal(0, 0.01, (self.embed_size, 1))),
				requires_grad = True) for j in range(self.vocab_size)] for i in range(2)]
		self.l_biases, self.r_biases = [
			[Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),
				requires_grad = True) for j in range(self.vocab_size)] for i in range(2)]

		self.l_embed.to(device)
		self.r_embed.to(device)
		self.l_biases.to(device)
		self.r_biases.to(device)
		# Set up optimizer
		self.optimizer = optim.Adam(self.l_embed + self.r_embed + self.l_biases + self.r_biases, lr = self.l_rate)


	# Weight function
	def wf(self,x):
		if x < self.xmax:
			return (x/self.xmax)**self.alpha
		return 1

	# Batch sampling function
	def gen_batch(self):
		sample = np.random.choice(np.arange(len(self.coocs)), size=self.batch_size, replace=False)
		l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [], [], [], [], []
		for chosen in sample:
			ind = tuple(self.coocs[chosen])
			l_vecs.append(self.l_embed[ind[0]])
			r_vecs.append(self.r_embed[ind[1]])
			covals.append(self.comat[ind])
			l_v_bias.append(self.l_biases[ind[0]])
			r_v_bias.append(self.r_biases[ind[1]])
		l_vecs = torch.tensor(l_vecs, device=device, dtype=torch.float64)
		r_vecs = torch.tensor(r_vecs, device=device, dtype=torch.float64)
		covals = torch.tensor(covals, device=device, dtype=torch.float64)
		l_v_bias = torch.tensor(l_v_bias, device=device, dtype=torch.float64)
		r_v_bias = torch.tensor(l_v_bias, device=device, dtype=torch.float64)
		return l_vecs, r_vecs, covals, l_v_bias, r_v_bias

	def train(self):
		# Train model
		for epoch in range(self.num_epochs):
			num_batches = int(self.w_list_size/self.batch_size)
			avg_loss = 0.0
			for batch in range(num_batches):
				self.optimizer.zero_grad()
				l_vecs, r_vecs, covals, l_v_bias, r_v_bias = self.gen_batch()
				loss = sum([torch.mul((torch.dot(l_vecs[i].view(-1), r_vecs[i].view(-1)) +
						l_v_bias[i] + r_v_bias[i] - np.log(covals[i]))**2,
						self.wf(covals[i])) for i in range(self.batch_size)])
				avg_loss += loss.data[0]/num_batches
				loss.backward()
				self.optimizer.step()
			print("Average loss for epoch "+str(epoch+1)+": ", avg_loss)

	def word_vec(self,word):
		if word not in self.w_to_i:
			return None
		return (self.l_embed[self.w_to_i[word]].data + self.r_embed[self.w_to_i[word]].data).numpy()
