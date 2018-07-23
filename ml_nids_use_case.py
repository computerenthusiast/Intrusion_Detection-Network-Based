import numpy as np
import pandas as pd
import csv as csv
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import zscore

from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import scikitplot as skplt
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from keras import optimizers

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name,x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis =1, inplace=True)


def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def encode_numeric_zscore(df, name, mean=None, sd= None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean)/sd
    

def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    #Encode to int for classification, float otherwise
    if target_type in (np.int64, np.int32):
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1, data_low= None, data_high = None):

    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low)/(data_high - data_low)) * (normalized_high- normalized_low) + normalized_low

print("Creating dataframe of train data.....")
df_train = pd.read_csv("C:/CNN_NEW/kddcup.data_10_percent_corrected")

df_test = pd.read_csv("C:/CNN_NEW/correctedDatasetnew")

print("Read {} rows.".format(len(df_train)))
print("Read {} rows.".format(len(df_test)))


df_train.dropna(inplace=True, axis = 1)

df_test.dropna(inplace=True, axis = 1)

df_train.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
              'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
              'num_comprised','root_shell','su_attempted','num_root','num_file_creations',
              'num_shells','num_access_files','num_outbounds_cmds','is_host_login','is_guest_login',
              'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
              'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
              'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
              'dst_host_rerror_rate','dst_host_srv_error_rate','outcome']

df_test.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
              'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
              'num_comprised','root_shell','su_attempted','num_root','num_file_creations',
              'num_shells','num_access_files','num_outbounds_cmds','is_host_login','is_guest_login',
              'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
              'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
              'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
              'dst_host_rerror_rate','dst_host_srv_error_rate','outcome']


df_train['outcome'][df_train['outcome'].str.contains('apache2|back|land|mailbomb|neptune|pod|processtable|smurf|teardrop|udpstorm|worm')]='DoS'
df_train['outcome'][df_train['outcome'].str.contains('ipsweep|mscan|nmap|portsweep|saint|satan')]='Probe'
df_train['outcome'][df_train['outcome'].str.contains('buffer_overflow|loadmodule|perl|rootkit|ps|sqlattack|xterm')]='U2R'
df_train['outcome'][df_train['outcome'].str.contains('ftp_write|guess_passwd|imap|multihop|phf|warezmaster|warezclient|xlock|xsnoop|snmpguess|snmpgetattack|httptunnel|sendmail|named|spy')]='R2L'

df_test['outcome'][df_test['outcome'].str.contains('apache2|back|land|mailbomb|neptune|pod|processtable|smurf|teardrop|udpstorm|worm')]='DoS'
df_test['outcome'][df_test['outcome'].str.contains('ipsweep|mscan|nmap|portsweep|saint|satan')]='Probe'
df_test['outcome'][df_test['outcome'].str.contains('buffer_overflow|loadmodule|perl|rootkit|ps|sqlattack|xterm')]='U2R'
df_test['outcome'][df_test['outcome'].str.contains('ftp_write|guess_passwd|imap|multihop|phf|warezmaster|warezclient|xlock|xsnoop|snmpguess|snmpgetattack|httptunnel|sendmail|named|spy')]='R2L'


encode_numeric_zscore(df_train, 'duration')
encode_numeric_range(df_train, 'duration')
encode_text_dummy(df_train, 'protocol_type')
#encode_numeric_range(df_train, 'protocol_type')
encode_text_dummy(df_train, 'service')
#encode_numeric_range(df_train, 'service')
encode_text_dummy(df_train, 'flag')
#encode_numeric_range(df_train, 'flag')
encode_numeric_zscore(df_train, 'src_bytes')
encode_numeric_range(df_train, 'src_bytes')
encode_numeric_zscore(df_train, 'dst_bytes')
encode_numeric_range(df_train, 'dst_bytes')
encode_text_dummy(df_train, 'land')
encode_numeric_zscore(df_train, 'wrong_fragment')
encode_numeric_zscore(df_train, 'urgent')
encode_numeric_zscore(df_train, 'hot')
encode_numeric_zscore(df_train, 'num_failed_logins')
encode_text_dummy(df_train, 'logged_in')
encode_numeric_zscore(df_train, 'num_comprised')
encode_numeric_zscore(df_train, 'root_shell')
encode_numeric_zscore(df_train, 'su_attempted')
encode_numeric_zscore(df_train, 'num_root')
encode_numeric_zscore(df_train, 'num_file_creations')
encode_numeric_zscore(df_train, 'num_shells')
encode_numeric_zscore(df_train, 'num_access_files')
encode_numeric_zscore(df_train, 'num_outbounds_cmds')
encode_text_dummy(df_train, 'is_host_login')
encode_text_dummy(df_train, 'is_guest_login')
encode_numeric_zscore(df_train, 'count')
encode_numeric_zscore(df_train, 'srv_count')
encode_numeric_zscore(df_train, 'serror_rate')
encode_numeric_zscore(df_train, 'srv_serror_rate')
encode_numeric_zscore(df_train, 'rerror_rate')
encode_numeric_zscore(df_train, 'srv_rerror_rate')
encode_numeric_zscore(df_train, 'same_srv_rate')
encode_numeric_zscore(df_train, 'srv_diff_host_rate')
encode_numeric_zscore(df_train, 'dst_host_count')
encode_numeric_zscore(df_train, 'dst_host_srv_count')
encode_numeric_zscore(df_train, 'dst_host_same_srv_rate')
encode_numeric_zscore(df_train, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df_train, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df_train, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df_train, 'dst_host_serror_rate')
encode_numeric_zscore(df_train, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df_train, 'dst_host_rerror_rate')
encode_numeric_zscore(df_train, 'dst_host_srv_error_rate')
outcome = encode_text_index(df_train, 'outcome')


######################################

encode_numeric_zscore(df_test, 'duration')
encode_numeric_range(df_test, 'duration')
encode_text_dummy(df_test, 'protocol_type')
encode_text_dummy(df_test, 'service')
encode_text_dummy(df_test, 'flag')
encode_numeric_zscore(df_test, 'src_bytes')
encode_numeric_range(df_test, 'src_bytes')
encode_numeric_zscore(df_test, 'dst_bytes')
encode_numeric_range(df_test, 'dst_bytes')
encode_text_dummy(df_test, 'land')
encode_numeric_zscore(df_test, 'wrong_fragment')
encode_numeric_zscore(df_test, 'urgent')
encode_numeric_zscore(df_test, 'hot')
encode_numeric_zscore(df_test, 'num_failed_logins')
encode_text_dummy(df_test, 'logged_in')
encode_numeric_zscore(df_test, 'num_comprised')
encode_numeric_zscore(df_test, 'root_shell')
encode_numeric_zscore(df_test, 'su_attempted')
encode_numeric_zscore(df_test, 'num_root')
encode_numeric_zscore(df_test, 'num_file_creations')
encode_numeric_zscore(df_test, 'num_shells')
encode_numeric_zscore(df_test, 'num_access_files')
encode_numeric_zscore(df_test, 'num_outbounds_cmds')
encode_text_dummy(df_test, 'is_host_login')
encode_text_dummy(df_test, 'is_guest_login')
encode_numeric_zscore(df_test, 'count')
encode_numeric_zscore(df_test, 'srv_count')
encode_numeric_zscore(df_test, 'serror_rate')
encode_numeric_zscore(df_test, 'srv_serror_rate')
encode_numeric_zscore(df_test, 'rerror_rate')
encode_numeric_zscore(df_test, 'srv_rerror_rate')
encode_numeric_zscore(df_test, 'same_srv_rate')
encode_numeric_zscore(df_test, 'srv_diff_host_rate')
encode_numeric_zscore(df_test, 'dst_host_count')
encode_numeric_zscore(df_test, 'dst_host_srv_count')
encode_numeric_zscore(df_test, 'dst_host_same_srv_rate')
encode_numeric_zscore(df_test, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df_test, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df_test, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df_test, 'dst_host_serror_rate')
encode_numeric_zscore(df_test, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df_test, 'dst_host_rerror_rate')
encode_numeric_zscore(df_test, 'dst_host_srv_error_rate')
outcome = encode_text_index(df_test, 'outcome')

num_classes = len(outcome)

df_train.dropna(inplace=True,axis=1)
df_test.dropna(inplace=True,axis=1)



x_train,y_train = to_xy(df_train, 'outcome')

x_test,y_test = to_xy(df_test, 'outcome')
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

print('model training starting....')
model = Sequential()
model.add(Convolution1D(64,3, input_shape=(x_train.shape[1],1), border_mode='same', activation='relu'))
model.add(Convolution1D(64,3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=(2)))
model.add(Convolution1D(128,3,border_mode='same', activation='relu'))
model.add(Convolution1D(128,3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=(2)))
model.add(Flatten())
model.add(Dense(128, activation='relu',W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax',W_regularizer=l2(0.001)))

#sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,patience=5,verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only= True)
model.fit(x_train,y_train, validation_data=(x_test,y_test), callbacks=[monitor,checkpointer], epochs=1)
#print(history.history.keys())

feat_train=model.predict(x_train)
feat_test=model.predict(x_test)
gnb=GaussianNB()
gnb.fit(feat_train,np.argmax(y_train,axis=1))
print("trainning score...",gnb.score(feat_train,np.argmax(y_train,axis=1)))
print("testing score...",gnb.score(feat_test,np.argmax(y_test,axis=1)))
pred_labels=gnb.predict(feat_test)
probas=gnb.predict_proba(feat_test)
confusion_matrix=metrics.confusion_matrix(np.argmax(y_test,axis=1),pred_labels)
print("\n\nConfusion Matrix {} %".format(confusion_matrix))
classification_report = metrics.classification_report(np.argmax(y_test,axis=1),pred_labels,target_names=outcome)
print("\n\nClassifiction Scores {} %".format(classification_report))
skplt.metrics.plot_precision_recall_curve(np.argmax(y_test,axis=1),probas)
plt.show()
skplt.metrics.plot_roc_curve(np.argmax(y_test,axis=1),probas)
plt.show()

