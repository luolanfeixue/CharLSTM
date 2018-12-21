import numpy as np
import copy
import pickle

def batch_generator(arr, n_seqs, n_steps):
	arr = copy.copy(arr)
	batch_size = n_seqs * n_steps
	n_batches = int(len(arr) / batch_size)
	arr = arr[:batch_size*n_batches]
	arr = arr.reshape((n_seqs, -1))
	print('arr',arr.shape)
	while True:
		np.random.shuffle(arr)
		for n in range(0, arr.shape[1], n_steps):
			x = arr[:, n : n + n_steps]
			y = np.zeros_like(x)
			# y[:, :-1] 将 x 的往后移动一位
			y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
			yield x, y
			


class TextConverter(object):
	def __init__(self,text, max_vocab=5000,char_vocab_file=None):
		if char_vocab_file is not None:
			with open(char_vocab_file, 'rb') as f:
				self.char_vocab = pickle.load(f)
		else:
			char_vocab = set(text)
			print(char_vocab)
			print('len of char vocab', len(char_vocab))
			# 计数
			char_vocab_count = {}
			for char in char_vocab:
				char_vocab_count[char] = 0
			for char in text:
				char_vocab_count[char] += 1
			# 转换 列表，用于排序。
			char_count_list = []
			for char in char_vocab_count:
				char_count_list.append((char,char_vocab_count[char]))
			char_count_list.sort(key=lambda x:x[1], reverse=True)
			if len(char_count_list) > max_vocab:
				char_count_list = char_count_list[:max_vocab]
			char_vocab = [x[0] for x in char_count_list]
			self.char_vocab = char_vocab
		
		self.char_to_idx_table = {c : i for i ,c in enumerate(self.char_vocab)}
		self.idx_to_char_table = {i : c for i, c in enumerate(self.char_vocab)}
	
	@property
	def vocab_size(self):
		return len(self.char_vocab) + 1

	def get_idx(self, char):
		if char in self.char_to_idx_table:
			return self.char_to_idx_table[char]
		else:
			return len(self.char_vocab)
	
	def get_char(self,index):
		if index < len(self.char_vocab):
			return self.idx_to_char_table[index]
		elif index == len(self.char_vocab):
			return '<unk>'
		else:
			raise Exception('Unknow index!')
	
	def get_arr(self, text):
		arr = []
		for char in text:
			arr.append(self.get_idx(char))
		return np.array(arr)
	
	def get_text(self, arr):
		chars = []
		for index in arr:
			chars.append(self.get_char(index))
		return "".join(chars)
	
	def save_to_file(self, char_vocab_file):
		with open(char_vocab_file, 'wb') as f:
			pickle.dump(self.char_vocab, f)