# encoding:utf-8
"""
@author = 'XXY'
@contact = '529379497@qq.com'
@researchFie1d = 'NLP DL ML'
@date= '2017/12/21 10:18'
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class TextCNN(object):
	def __init__(self, sequence_length_char, sequence_length_punc, num_classes, vocab_size_char, vocab_size_punc,
				 embedding_size, filter_sizes_char, filter_sizes_punc, num_filters, l2_reg_lambda=0.0):
		'''
		:param sequence_length: 表示文本长度，多少个词
		:param num_classes: 	待分类的类别个数
		:param vocab_size: 		词库的大小，表示构建的词库有多大
		:param embedding_size:  词向量维度大小
		:param filter_sizes:	卷积核的尺寸，是一个列表的形式[1,2,3]
		:param num_filters:		卷积核的个数
		:param l2_reg_lambda:	正则化系数
		'''

		with tf.name_scope('input'):	# 一个输入的命名空间
			self.input_x_char = tf.placeholder(tf.int32, [None, sequence_length_char], name='input_x_char')
			self.input_x_punc = tf.placeholder(tf.int32, [None, sequence_length_punc], name='input_x_punc')
			self.input_x_fc_feat = tf.placeholder(tf.float32, [None, None], name='input_x_fc_feat')
			self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

		with tf.name_scope('dropout'):
			self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		# with tf.device('/gpu:0'), tf.name_scope('embedding'):
		with tf.name_scope('embedding-char'):
			W = tf.Variable(tf.random_uniform([vocab_size_char, embedding_size], -1.0, 1.0), name='W')
			# tf.summary.histogram('embedding',W)							#这个是tensorboard画图的
			self.embedded_char = tf.nn.embedding_lookup(W, self.input_x_char)
			self.embedded_char_expanded = tf.expand_dims(self.embedded_char, -1)

		with tf.name_scope('embedding-punc'):
			W = tf.Variable(tf.random_uniform([vocab_size_punc, embedding_size], -1.0, 1.0), name='W')
			# tf.summary.histogram('embedding',W)							#这个是tensorboard画图的
			self.embedded_punc = tf.nn.embedding_lookup(W, self.input_x_punc)
			self.embedded_punc_expanded = tf.expand_dims(self.embedded_punc, -1)


			#增加一个维度，变成batch_size*seq_len*em_size*channel(=1)的4维tensor,符合图像的习惯

		# Create a convolution + maxpool layer for each filter size

		pooled_outputs_char = []
		pooled_outputs_punc = []
		for i, filter_size_char in enumerate(filter_sizes_char):#比如（0,3），（1,4），（2,5）
			with tf.name_scope('conv-char-maxpool-%s' % filter_size_char):				    # 循环一次建立一个名称为如“conv-ma-3”的模块
				# Convolution Layer
				filter_shape = [filter_size_char, embedding_size, 1, num_filters]	    #卷积核的参数，[高，宽，通道数，卷积核个数]
				W_char = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_char')	#卷积核的初始化

				# tf.summary.histogram('convW-%s' % filter_size, W)				    #tensorboard画图
				b_char = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b_char')    # 偏置b，维度为卷积核个数的tensor
				# tf.summary.histogram('convb-%s' % filter_size,b)				    #tensorboard画图
				conv_char = tf.nn.conv2d(				#卷积运算
					self.embedded_char_expanded,	#输入特征矩阵
					W_char,								#初始化的卷积核矩阵
					strides=[1, 1, 1, 1],			#划窗移动距离[1, 横向距离， 纵向距离, 1]
					padding='VALID',				#边缘是否补0
					name='conv_char')

				h_char = tf.nn.relu(tf.nn.bias_add(conv_char, b_char), name='relu')	#卷积之后使用relu（）激活函数去线性化

				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(			#池化运算
					h_char,								#卷积后的输入矩阵
					ksize=[1, sequence_length_char - filter_size_char + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool_char')
				pooled_outputs_char.append(pooled)

		for i, filter_size in enumerate(filter_sizes_punc):  # 比如（0,3），（1,4），（2,5）
			with tf.name_scope('conv-punc-maxpool-%s' % filter_size):				    # 循环一次建立一个名称为如“conv-ma-3”的模块
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]	    #卷积核的参数，[高，宽，通道数，卷积核个数]
				W_punc = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_punc')	#卷积核的初始化
				# tf.summary.histogram('convW-%s' % filter_size, W)				    #tensorboard画图
				b_punc = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b_punc')    # 偏置b，维度为卷积核个数的tensor
				# tf.summary.histogram('convb-%s' % filter_size,b)				    #tensorboard画图
				conv_punc = tf.nn.conv2d(				#卷积运算
					self.embedded_punc_expanded,	#输入特征矩阵
					W_punc,								#初始化的卷积核矩阵
					strides=[1, 1, 1, 1],			#划窗移动距离[1, 横向距离， 纵向距离, 1]
					padding='VALID',				#边缘是否补0
					name='conv_punc')
				h_punc = tf.nn.relu(tf.nn.bias_add(conv_punc, b_punc), name='relu')	#卷积之后使用relu（）激活函数去线性化

				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(			#池化运算
					h_punc,								#卷积后的输入矩阵
					ksize=[1, sequence_length_punc - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool_punc')
				pooled_outputs_punc.append(pooled)

		# with tf.name_scope('lstm'):
		# 	lstm_cell = rnn.BasicLSTMCell(num_units=128, forget_bias=1.0, state_is_tuple=True)
		#
		# 	# **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
		# 	lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
		#
		# 	# **步骤4：调用 MultiRNNCell 来实现多层 LSTM
		# 	mlstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
		#
		# 	# **步骤5：用全零来初始化state
		# 	init_state = mlstm_cell.zero_state(tf.placeholder(tf.int32, name='init_state'), dtype=tf.float32)
		# 	#
		# 	# #**步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
		# 	# # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
		# 	# # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
		# 	# # ** state.shape = [layer_num, 2, batch_size, hidden_size],
		# 	# # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
		# 	# # ** 最后输出维度是 [batch_size, hidden_size]
		# 	outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=self.embedded_punc, initial_state=init_state, time_major=False)
		# 	h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
		#
		# 	# *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
		# 	# 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
		# 	# **步骤6：方法二，按时间步展开计算
		# 	# outputs = list()
		# 	# state = init_state
		# 	# with tf.variable_scope('RNN'):
		# 	# 	for timestep in range(timestep_size):
		# 	# 		if timestep > 0:
		# 	# 			tf.get_variable_scope().reuse_variables()
		# 	# 		# 这里的state保存了每一层 LSTM 的状态
		# 	# 		(cell_output, state) = mlstm_cell(X[:, timestep, :], state)
		# 	# 		outputs.append(cell_output)
		# 	# h_state = outputs[-1]
		#
		# 	# with tf.name_scope("lstm"):
		# 	# 	lstm_cell = tf.contrib.rnn.BasicLSTMCell(128)
		# 	# 	init_state = lstm_cell.zero_state(batch_size=, dtype=tf.float32)
		# 	# 	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, self.input_x_punc, initial_state=init_state, time_major=False)
		# 	# 	# results = tf.matmul(final_state[1], weights['out']) + biases['out']
		num_filters_total_char = num_filters * len(filter_sizes_char)                   #每种卷积核个数与卷积种类的积
		num_filters_total_punc = num_filters * len(filter_sizes_punc)					#每种卷积核个数与卷积种类的积

		self.h_pool_char = tf.concat(pooled_outputs_char,3)	# 将outputs在第4个维度上拼接，如本来是128*1*1*64的结果3个，拼接后为128*1*1*192的tensor
		self.h_pool_flat_char = tf.reshape(self.h_pool_char, [-1, num_filters_total_char]) #  将最后结果reshape为128*192的tensor

		self.h_pool_punc = tf.concat(pooled_outputs_punc,3)  # 将outputs在第4个维度上拼接，如本来是128*1*1*64的结果3个，拼接后为128*1*1*192的tensor
		self.h_pool_flat_punc = tf.reshape(self.h_pool_punc, [-1, num_filters_total_punc])  # 将最后结果reshape为128*192的tensor

		self.fc_vec = tf.concat([self.h_pool_flat_char, self.h_pool_flat_punc], 1)

		with tf.name_scope('dense_layer1'):
			W_dense1 = tf.get_variable(
				'W_dense1',
				shape=[num_filters_total_char + num_filters_total_punc, 256],
				initializer=tf.contrib.layers.xavier_initializer())
			b_dense1 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_dense1')
			self.dense1 = tf.nn.xw_plus_b(self.fc_vec, W_dense1, b_dense1, name='dense1')
			self.dense_feat = tf.concat([self.dense1, self.input_x_fc_feat], 1)

		# Add dropout
		with tf.name_scope('dropout1'):				# 添加一个"dropout"的模块，里面一个操作，输出为dropout过后的128*192的tensor
			self.h_drop_1 = tf.nn.dropout(self.dense_feat, self.dropout_keep_prob)
		#
		# with tf.name_scope('dense_layer2'):
		# 	W_dense2 = tf.get_variable(
		# 		'W_dense2',
		# 		shape=[256, 64],
		# 		initializer=tf.contrib.layers.xavier_initializer())
		# 	b_dense2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b')
		# 	self.dense2 = tf.nn.xw_plus_b(self.dense1, W_dense2, b_dense2, name='dense1')

		# # Add dropout
		# with tf.name_scope('dropout2'):				# 添加一个"dropout"的模块，里面一个操作，输出为dropout过后的128*192的tensor
		# 	self.h_drop_2 = tf.nn.dropout(self.dense2, self.dropout_keep_prob)		# 使用dropout机制防止过拟合


		# Final (unnormalized) scores and predictions
		with tf.name_scope('output'):	#全连接操作，到输出层，注意这里用的是get_variables
			W_output = tf.Variable(
				tf.random_normal([256+200, num_classes], stddev=0.35),
				name = "weights"
			)
			b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')		#输出层的偏置
			l2_loss += tf.nn.l2_loss(W_output)				                            #对全连接层的W使用l2_loss正则
			l2_loss += tf.nn.l2_loss(b_output)				                            #对全连接层的b使用l2_loss正则
			self.scores = tf.nn.xw_plus_b(self.h_drop_1, W_output, b_output, name='scores')# 相当于tf.nn.matmul(self.h_drop, W) + b
			self.predictions = tf.argmax(self.scores, 1, name='predictions')		     # 转换成one-hot的编码形式

		# Calculate mean cross-entropy loss
		with tf.name_scope('loss'):#定义一个”loss“的模块
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores) #  交叉熵损失函数
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss	#计算loss（包含正则化系数）
			# tf.summary.scalar('loss',self.loss)	#tensorboard 画图形式

		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
			# operation2，计算均值即为准确率，名称”accuracy“

		with tf.name_scope('correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')