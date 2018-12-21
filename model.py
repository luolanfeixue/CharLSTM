from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


def pick_top_n(preds, vocab_size, top_n=5):
	p = np.squeeze(preds)
	# 将除了top_n个预测值的位置都置为0
	p[np.argsort(p)[:-top_n]] = 0
	# 归一化概率
	p = p / np.sum(p)
	# 随机选取一个字符
	c = np.random.choice(vocab_size, 1, p=p)[0]
	return c


class CharRNN:
	def __init__(self, num_classes, num_seqs=64, num_steps=50,
	             lstm_size=128, num_layers=2, learning_rate=0.001,
	             grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
		"""

		:rtype: object
		"""
		if sampling is True:
			num_seqs, num_steps = 1, 1
		else:
			num_seqs, num_steps = num_seqs, num_steps
		
		self.num_classes = num_classes
		self.num_seqs = num_seqs
		self.num_steps = num_steps
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.learning_rate = learning_rate
		self.grad_clip = grad_clip
		self.train_keep_prob = train_keep_prob
		self.use_embedding = use_embedding
		self.embedding_size = embedding_size
		
	
	def build_model(self):
		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(tf.int32, shape=(
				self.num_seqs, self.num_steps), name='inputs')
			self.labels = tf.placeholder(tf.int32, shape=(
				self.num_seqs, self.num_steps), name='labels')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		if not self.use_embedding:
			self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
		else:
			with tf.device("/cpu:0"):
				embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
				self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
				
		def get_a_call(lstm_size, keep_prob):
			lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
			drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
			return drop
		
		with tf.name_scope('lstm'):
			# 组织多层lstm
			cell = tf.nn.rnn_cell.MultiRNNCell([get_a_call(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)])
			# 将lstm在时间维度上展开
			self.initial_state = cell.zero_state(self.num_seqs, tf.float32)
			self.lstm_output, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)
			
			#通过lstm_outputs得到概率
			self.seq_output = tf.concat(self.lstm_output, 1)
			self.x = tf.reshape(self.seq_output, [-1, self.lstm_size])
			
			with tf.variable_scope('softmax'):
				softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
				softmax_b = tf.Variable(tf.zeros(self.num_classes))
			
			self.logits = tf.matmul(self.x, softmax_w) + softmax_b
			self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')
			
		with tf.name_scope('loss'):
			y_one_hot = tf.one_hot(self.labels, self.num_classes)
			y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
			self.loss = tf.reduce_mean(loss)
		
		self.add_train_op('adam', self.learning_rate, self.loss, self.grad_clip)
		self.initialize_session()
	
	def add_train_op(self, lr_method, lr, loss, clip):
		_lr_m = lr_method.lower()
		with tf.variable_scope('train_step'):
			if _lr_m == 'adam':
				optimizer = tf.train.AdamOptimizer(lr)
			elif _lr_m == 'adagrad':
				optimizer = tf.train.AdagradDAOptimizer(lr)
			elif _lr_m == 'sgd':
				optimizer = tf.train.GradientDescentOptimizer(lr)
			elif _lr_m == 'rmsprob':
				optimizer = tf.train.RMSPropOptimizer()
			else:
				raise NotImplementedError('Unknown methond {}'.format(_lr_m))
			
			if clip > 0:
				grads, vs = zip(*optimizer.compute_gradients(loss))
				grads, gnorm = tf.clip_by_global_norm(grads, clip)
				self.train_op = optimizer.apply_gradients(zip(grads, vs))
			else:
				self.train_op = optimizer.minimize(loss)
				
	def initialize_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		
	def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
		step = 0
		new_state = self.sess.run(self.initial_state)
		for x, y in batch_generator:
			step += 1
			start = time.time()
			feed = {self.inputs:x, self.labels: y, self.keep_prob:self.train_keep_prob, self.initial_state: new_state}
			batch_loss,new_state, _ = self.sess.run([self.loss,self.final_state,self.train_op], feed_dict=feed)
			end = time.time()
			if step % log_every_n == 0:
				print('step: {}/{}... '.format(step, max_steps),
				      'loss: {:.4f}... '.format(batch_loss),
				      '{:.4f} sec/batch'.format((end - start)))
			if (step % save_every_n == 0):
				self.saver.save(self.sess, os.path.join(save_path,'model'), global_step=step)
			if step >= max_steps:
				break
		self.saver.save(self.sess, os.path.join(save_path, 'model'), global_step=step)
		
	
	def sample(self, n_samples, prime, vocab_size):
		samples = [c for c in prime]
		sess = self.session
		new_state = sess.run(self.initial_state)
		preds = np.ones((vocab_size,))  # for prime=[]
		for c in prime:
			x = np.zeros((1, 1))
			# 输入单个字符
			x[0, 0] = c
			feed = {self.inputs: x,
			        self.keep_prob: 1.,
			        self.initial_state: new_state}
			preds, new_state = sess.run([self.proba_prediction, self.final_state],
			                            feed_dict=feed)
		
		c = pick_top_n(preds, vocab_size)
		# 添加字符到samples中
		samples.append(c)
		
		# 不断生成字符，直到达到指定数目
		for i in range(n_samples):
			x = np.zeros((1, 1))
			x[0, 0] = c
			feed = {self.inputs: x,
			        self.keep_prob: 1.,
			        self.initial_state: new_state}
			preds, new_state = sess.run([self.proba_prediction, self.final_state],
			                            feed_dict=feed)
			
			c = pick_top_n(preds, vocab_size)
			samples.append(c)
		
		return np.array(samples)

	def load(self, checkpoint):
		self.session = tf.Session()
		self.saver.restore(self.session, checkpoint)
		print('Restored from: {}'.format(checkpoint))