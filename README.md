# SMP2018-task1
深度学习用于今日日头条用户画像

相关文件说明
<p>-- config/     网络结构的参数配置
<p>-- data/       原数据，参考SMP2018官网数据下载
<p>-- models/     训练的模型保存文件夹
<p>-- results/    最后的输出结果保存文件夹


<p>-- data_helper.py 数据预处理过程，生成相应的文件
<p>-- cnn_char.py  以字为单位的cnn网络结构
<p>-- cnn_char_predict.py 以字（char）为单位的cnn在测试集上的预测结果
<p>-- cnn_char_train.py   以字（char）为单位的cnn在训练集上的训练过程
<p>-- cnn_char_punc.py    以字为单位并增加标点符号等的其他特征的cnn网络结构
<p>-- cnn_char_punc_train.py    以字为单位并增加标点符号等的其他特征在训练集上的训练过程
<p>-- cnn_char_punc_predict.py    以字为单位并增加标点符号等的在测试集上的预测结果

<p>--lstm_example.py    lstm实现分类在手写数字识别上的简单实例


<font size =5>机器学习方法<font/>
<p>--svm.py   使用机器学习方法SVM的训练过程     
<p>--predict.py  使用SVM的预测过程
