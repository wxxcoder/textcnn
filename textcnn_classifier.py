

from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gensim
from keras.layers import Dense,Input,Flatten
from keras.layers import Reshape,Dropout,Concatenate
from keras.layers import Conv2D,MaxPool2D,Embedding
from keras.models import Model
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from keras.models import load_model
import jieba

MAX_SEQUENCE_LENGTH = 50 # Maximum number of words in a sentence
MAX_NB_WORDS = 20000 # Vocabulary size
EMBEDDING_DIM = 300 # Dimensions of Glove word vectors
VALIDATION_SPLIT = 0.1

class TextCNNClassifier(object):

    def __init__(self,label_encoder=None,clf=None):
        if label_encoder is not None:
            self.label_encoder=label_encoder
        else:
            self.label_encoder=LabelEncoder()
        self.clf=clf
        self.label_dict={}
        self.word_index={}
        self.embeddings_matrix=None
        self.tokenizer=None


    @classmethod
    def required_packages(cls):
        return ["sklearn","numpy","keras","gensim","tensorflow"]


    def label_fit(self,label_path):
        "对label_encoder先使用所有的label数据fit以下，后面只要transform就可以"
        labels=np.load(label_path).tolist()
        return self.label_encoder.fit(labels)

    def label_str2num(self,labels:list)->np.ndarray:
        '''将文本对应的标签从字符串转化为数字，使用LabelEncoder'''
        return self.label_encoder.transform(labels)

    def label_num2str(self,y:np.ndarray)->np.ndarray:
        '''将文本对应的标签从数字转化为字符串，使用LabelEncoder'''
        return self.label_encoder.inverse_transform(y)

    def train(self,x_train,y_train,x_val,y_val,batch_size=64,epochs=5):
        '''对该模型进行训练'''

        callbacks_list=[
            EarlyStopping( #对val_acc这一评价指标进行监测，如果超过2个epoch性能没有提升，则提前停止训练
                monitor='val_acc',
                patience=2,
                verbose=1,
                mode='auto'
            ),
            ModelCheckpoint(#以val_loss为评价指标，保存最佳的模型
                filepath='model/my_model.h5',
                monitor='val_acc',
                save_best_only=True,
            )
        ]
        self.clf=self._create_classifier()
        self.clf.fit(x_train,y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=callbacks_list,
                     validation_data=(x_val,y_val),
                     verbose=2)


    def eval_model(y_true, y_pred, labels=None):
        '''计算每个分类的Precision, Recall, f1, support'''
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
        # 计算总体的平均Precision, Recall, f1, support
        tot_p = np.average(p, weights=s)
        tot_r = np.average(r, weights=s)
        tot_f1 = np.average(f1, weights=s)
        tot_s = np.sum(s)
        res1 = pd.DataFrame({
            u'Label': labels,
            u'Precision': p,
            u'Recall': r,
            u'F1': f1,
            u'Support': s
        })
        res2 = pd.DataFrame({
            u'Label': ['总体'],
            u'Precision': [tot_p],
            u'Recall': [tot_r],
            u'F1': [tot_f1],
            u'Support': [tot_s]
        })
        res2.index = [999]
        res = pd.concat([res1, res2])
        return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]


    def _create_classifier(self,filter_sizes = [2, 3, 5],num_filters = 512,drop = 0.5):
        '''textCNN模型，这三个参数先用默认值吧，后续模型优化时再修改'''
        print("Creating Model...")
        inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding = Embedding(input_dim=len(self.word_index) + 1,
                              output_dim=EMBEDDING_DIM,
                              weights=[self.embeddings_matrix],
                              input_length=MAX_SEQUENCE_LENGTH,
                              trainable=True)(inputs)
        reshape = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedding)

        conv_0 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[0], EMBEDDING_DIM),
                        padding='valid',
                        kernel_initializer='normal',
                        activation='relu')(reshape)
        conv_1 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[1], EMBEDDING_DIM),
                        padding='valid',
                        kernel_initializer='normal',
                        activation='relu')(reshape)
        conv_2 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[2],
                                     EMBEDDING_DIM),
                        padding='valid',
                        kernel_initializer='normal',
                        activation='relu')(reshape)
        maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)

        dropout = Dropout(drop)(flatten)

        preds = Dense(len(self.label_dict),
                      activation='softmax')(dropout)

        # this creates a model that includes inputs and outputs
        model = Model(inputs=inputs, outputs=preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        model.summary()
        return model


    def predict(self,X:np.ndarray):
        '''对于一个文本数组（特征化后），返回每一个文本肯能行最大的标签
                    '''
        return self.clf.predict(X)

    def get_data(self,text_path,labels_path):
        '''获取训练集与验证集'''
        print("Preparing training and validation data")
        texts=np.load(open(text_path,'rb'),allow_pickle=True).tolist()
        labels=np.load(open(labels_path,'rb'),allow_pickle=True).tolist()
        labels=self.label_str2num(labels)
        label_dict = dict(zip(list(self.label_encoder.classes_),
                              self.label_encoder.transform(list(self.label_encoder.classes_))))
        self.label_dict = label_dict#获取了数据中的label_id的dict

        tokenizer=Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences=tokenizer.texts_to_sequences(texts)
        word_index=tokenizer.word_index
        self.tokenizer=tokenizer
        self.word_index=word_index
        print("Found %s unique tokens." % len(word_index))#得到数据中不重复词的个数
        data=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)#对文本长度进行填充，使得格式一致，此时data中的是一系列数字，词与数字的映射存放在word_index中
        labels=to_categorical(np.asarray(labels))

        #将数据分为训练集与验证集
        indices=np.arange(data.shape[0])
        np.random.shuffle(indices)
        data=data[indices]
        labels=labels[indices]
        num_validation_sample=int(VALIDATION_SPLIT*data.shape[0])
        x_train=data[:-num_validation_sample]
        y_train=labels[:-num_validation_sample]

        x_val=data[-num_validation_sample:]
        y_val=labels[-num_validation_sample:]

        return x_train,y_train,x_val,y_val

    def get_test_data(self,test_text_path,test_label_path):
        print("Preparing testing data")
        test_texts=np.load(open(test_text_path,'rb'),allow_pickle=True).tolist()
        test_labels=np.load(open(test_label_path,'rb'),allow_pickle=True).tolist()
        test_sequences=self.tokenizer.texts_to_sequences(test_texts)
        test_input=pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)
        return test_input,test_labels


    def prepare_embeddings_matrix(self,embedding_path):
        '''加载已经训练好的词向量，得到词嵌入矩阵'''
        print('Preparing embedding matrix.')
        embeddings_index=self._load_embedding(embedding_path)
        num_words=min(MAX_NB_WORDS,len(self.word_index))
        embeddings_matrix=np.zeros((len(self.word_index)+1,EMBEDDING_DIM))
        count=0
        for word,i in self.word_index.items():
            embeddings_vectors=embeddings_index.get(word)
            if embeddings_vectors is not None:
                count+=1
                embeddings_matrix[i]=embeddings_vectors
        self.embeddings_matrix=embeddings_matrix
        print("能找到%s个存在的向量" % count)
        return embeddings_matrix


    def _load_embedding(self,embedding_path):
        '''加载词向量，返回词：向量的dict'''
        embeddings_index={}
        model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        for word in model.vocab:
            coefs = model.get_vector(word)
            embeddings_index[word] = coefs
        print('This embedding_path found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def persist(self,model_path,model_name):
        '''保存该模型'''
        print("正在将模型保存到"+model_path + model_name + '.h5')
        self.clf.save(model_path + model_name + '.h5')

    def load(self,model_path):
        '''后续要使用时加载该模型'''
        return load_model(model_path)

    def process(self,message:str):
        '''对给定的文本返回可能性最大大的意图'''
        seq_list = jieba.cut(message, cut_all=False)
        text_cut = ' '.join(seq_list)
        print(text_cut)
        test_texts = [text_cut]
        test_sequences=self.tokenizer.texts_to_sequences(test_texts)
        test_input = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        test_predictions_probas = self.clf.predict(test_input)
        test_predictions = test_predictions_probas.argmax(axis=-1)
        print(test_predictions)
        print(self.label_num2str(test_predictions))
        return self.label_num2str(test_predictions)
