# encoding:utf-8
"""
@author = 'XXY'
@contact = '529379497@qq.com'
@researchFie1d = 'NLP DL ML'
@date= '2017/12/21 10:18'
"""
import os
import json
import time
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from cnn_char_punc import TextCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_helper import batch_iter

logging.getLogger().setLevel(logging.INFO)


def train():
	X_char = []
	X_punc = []
	y = []
	word2id = json.loads(open('./data/word2id.json').read())
	punc2id = json.loads(open('./data/punc2id.json').read())

	print "读取char特征"
	with open('./data/training_char.txt') as f:
		for line in f:
			temp = line.strip().split('\t')
			text = temp[0][1:-1].split(',')
			label = temp[1]
			X_char.append(text)
			y.append(label)

	print "读取punc特征"
	with open('./data/training_punc.txt') as f:
		for line in f:
			temp = line.strip().split('\t')
			text = temp[0][1:-1].split(',')
			X_punc.append(text)
	print "读取全连接层特征"
	X_df_fc = pd.read_csv('./data/training_fc_feat_norm_200.txt', sep=',')

	print"数据加载完毕！"

	labels = sorted(list(set(y)))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))
	y = [label_dict[i] for i in y]

	X_char = np.array(X_char)
	X_punc = np.array(X_punc)
	X_fc_feat = X_df_fc.values
	y = np.array(y)

	parameter_file = './config/cnn_parameters.json'
	params = json.loads(open(parameter_file).read())
	print"所有训练数据的大小：", X_char.shape, X_punc.shape
	X_train_char, X_dev_char, X_train_punc, X_dev_punc, X_train_fc_feat, X_dev_fc_feat, y_train, y_dev = \
		train_test_split(X_char, X_punc, X_fc_feat, y, random_state=10, test_size=0.1)
	print"训练数据大小：", X_train_char.shape, X_train_punc.shape, X_train_fc_feat.shape
	print"验证数据大小：", X_dev_char.shape, X_dev_punc.shape, X_dev_fc_feat.shape

	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				sequence_length_char = X_char.shape[1],
				sequence_length_punc = X_punc.shape[1],
				num_classes=len(labels),
				vocab_size_char=len(word2id),
				vocab_size_punc=len(punc2id),
				embedding_size=params['embedding_dim'],
				filter_sizes_char=list(map(int, params['filter_sizes_char'].split(","))),
				filter_sizes_punc=list(map(int, params['filter_sizes_punc'].split(","))),
				num_filters=params['num_filters'],
				l2_reg_lambda=params['l2_reg_lambda']
			)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join("models", "cnn_models", "trained_model_" + timestamp))

			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables())


			def train_step(input_x_char, input_x_punc, input_x_fc_feat, y_train):
				feed_dict = {
					cnn.input_x_char: input_x_char,
					cnn.input_x_punc: input_x_punc,
					cnn.input_x_fc_feat: input_x_fc_feat,
					cnn.input_y: y_train,
					cnn.dropout_keep_prob: params['dropout_keep_prob']
				}
				_, step, loss, acc, prediction = sess.run([train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
				print("After training {} step loss is: {}, accuracy is {}".format(step, loss, acc))

			def test_step(input_x_char, input_x_punc, input_x_fc_feat, y_test):
				feed_dict = {
					cnn.input_x_char: input_x_char,
					cnn.input_x_punc: input_x_punc,
					cnn.input_x_fc_feat: input_x_fc_feat,
					cnn.input_y: y_test,
					cnn.dropout_keep_prob: params['dropout_keep_prob']
				}
				step, loss, acc, num_correct, prediction = sess.run(
					[global_step, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions], feed_dict)
				return num_correct, prediction, loss, acc


			# 下面开始训练过程
			# Save the word_to_id map since predict.py needs it

			sess.run(tf.global_variables_initializer())
			# 对训练集分batch
			train_batches = batch_iter(zip(X_train_char, X_train_punc, X_train_fc_feat, y_train), params['batch_size'], params['num_epochs'])

			X_dev = zip(X_dev_char, X_dev_punc, X_dev_fc_feat, y_dev)
			best_accuracy, best_at_step = 0, 0

			for train_batch in train_batches:
				'''
				对labels（y_test_batches）进行one-hot编码操作
				'''
				X_train_char_batch, X_train_punc_batch, X_train_fc_feat_batch, y_train_batch = zip(*train_batch)
				train_step(X_train_char_batch, X_train_punc_batch, X_train_fc_feat_batch, y_train_batch)
				current_step = tf.train.global_step(sess, global_step)
				if current_step % params['evaluate_every'] == 0:  # 多少步评估一次
					total_dev_correct = 0
					dev_predictions = []

					for i in range(int(len(X_dev) / params['batch_size']) + 1):
						start_index = i * params['batch_size']
						end_index = min((i + 1) * params['batch_size'], len(X_dev))
						X_dev_batch = X_dev[start_index: end_index]
						X_dev_batch_char, X_dev_batch_punc, X_dev_batch_fc_feat, y_test_batch = zip(*X_dev_batch)
						num_dev_correct, dev_prediction, loss, acc = test_step(X_dev_batch_char, X_dev_batch_punc, X_dev_batch_fc_feat, y_test_batch)
						total_dev_correct += num_dev_correct
						dev_predictions = np.concatenate([dev_predictions, dev_prediction])
					print "最后预测结果：", dev_predictions
					print "长度为：", len(dev_predictions)
					print "最后预测结果：", dev_predictions
					dev_accuracy = float(total_dev_correct) / len(y_dev)
					logging.critical('Loss on dev set is:{}, Accuracy on dev set: {}'.format(loss, dev_accuracy))
					if dev_accuracy >= best_accuracy:
						best_accuracy, best_at_step = dev_accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
						logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))

if __name__ == '__main__':
	train()
