# encoding:utf-8
"""
@author = 'XXY'
@contact = '529379497@qq.com'
@researchFie1d = 'NLP DL ML'
@date= '2017/12/21 10:18'
"""
import json, os
import jieba.posseg as pseg
import logging
from data_helper import *
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import classification_report, accuracy_score

logging.getLogger().setLevel(logging.INFO)


def make_submission(file, prediction, encoding):
	valid_id = []
	label2int = {u"人类作者": 0, u"机器作者": 1, u"机器翻译": 2, u"自动摘要": 3}
	int2label = {0.0: u"人类作者", 1.0: u"机器作者", 2.0: u"机器翻译", 3.0: u"自动摘要"}

	for line in open('./data/validation.txt'):
		text = json.loads(line.strip())
		valid_id.append(text['id'])
	result = pd.DataFrame({'id': valid_id, 'label': prediction})
	result['label'] = result['label'].apply(lambda x: int2label[x])
	print(result.head())
	result.to_csv(file, header=None, index=None, encoding=encoding)


def predict_unseen_data():
	X_char = []
	X_punc = []
	y = []

	# 读取字符特征
	print "读取char特征"
	with open('./data/validation_char.txt') as f:
		for line in f:
			temp = line.strip().split('\t')
			text = temp[0][1:-1].split(',')
			label = temp[1]
			X_char.append(text)
			y.append(label)

	# 读取标点符号结构特征
	print "读取punc特征"
	with open('./data/validation_punc.txt') as f:
		for line in f:
			temp = line.strip().split('\t')
			text = temp[0][1:-1].split(',')
			X_punc.append(text)

	print "读取全连接层特征"
	X_validation_fc_feat = pd.read_csv('./data/validation_fc_feat_norm_200.txt', sep=',')
	print"数据加载完毕！"


	X_validation_char = np.array(X_char)
	X_validation_punc = np.array(X_punc)
	X_validation_fc_feat = X_validation_fc_feat.values
	params = json.loads(open('./config/cnn_parameters.json').read())
	print "validation数据大小：", X_validation_char.shape, X_validation_punc.shape
	print "validation集加载完毕！"

	checkpoint_dir = 'models/cnn_models/trained_model_1528419951/'
	if not checkpoint_dir.endswith('/'):
		checkpoint_dir += '/'
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')  # 加载最近保存的模型
	# checkpoint_file = '/home/h325/data/Xxy/SMP_NEW/models/cnn_models/trained_model_1528257857/checkpoints/model-12400'
	logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)

		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			input_x_char = graph.get_operation_by_name("input/input_x_char").outputs[0]
			input_x_punc = graph.get_operation_by_name("input/input_x_punc").outputs[0]
			input_x_fc_feat = graph.get_operation_by_name('input/input_x_fc_feat').outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout/dropout_keep_prob").outputs[0]
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]

			dev_predictions = []
			for i in range(int(len(X_validation_char) / params['batch_size']) + 1):
				start_index = i * params['batch_size']
				end_index = min((i + 1) * params['batch_size'], len(X_validation_char))
				X_validation_char_batch = X_validation_char[start_index: end_index]
				X_validation_punc_batch = X_validation_punc[start_index: end_index]
				X_validation_fc_feat_batch = X_validation_fc_feat[start_index: end_index]
				prediction= sess.run(predictions, {input_x_char: X_validation_char_batch, input_x_punc:X_validation_punc_batch,
				                                   input_x_fc_feat:X_validation_fc_feat_batch, dropout_keep_prob: 1.0})
				dev_predictions = np.concatenate([dev_predictions, prediction])

			# for test_batch in test_batches:
			# 	X_test_batch = test_batch
			# 	prediction= sess.run(predictions, {input_x1: X_test_batch, dropout_keep_prob: 1.0})
			# 	dev_predictions = np.concatenate([dev_predictions, prediction])
			make_submission('results/cnn_char_punc_fc_feat_result4.csv', dev_predictions, encoding='utf-8')
			logging.critical('The prediction is complete')
if __name__ == '__main__':
	predict_unseen_data()
