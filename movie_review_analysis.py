import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from training_data import create_input_data
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
train, test = create_input_data('pos.txt', 'neg.txt')

size_data = 100
x_train = [train[i][0] for i in range(len(train))]
y_train = [train[i][1] for i in range(len(train))]

x_test = [test[i][0] for i in range(len(test))]
y_test = [test[i][1] for i in range(len(test))]
num_words=2000
print('loaded data!!')
print(x_train[0])
print(y_train[0])
data_set= x_train + x_test

tokenizer=Tokenizer(num_words)
tokenizer.fit_on_texts(data_set)
#Each word in now mapped to a number

x_train_token=tokenizer.texts_to_sequences(x_train)
x_test_token=tokenizer.texts_to_sequences(x_test)

#padding the dataset

data_len=[len(x) for x in x_train_token]
mean_size=np.mean(data_len)#mean_size
maximum_allowable_length = int(2*mean_size)

x_train_padded = np.array([np.array(([0]*(int(maximum_allowable_length - data_len[i]))+x_train_token[i])
                           if (maximum_allowable_length - data_len[i])>0
                           else x_train_token[i][0:maximum_allowable_length])
                  for i in range(len(x_train))])
x_test_padded = np.array([[0]*int((maximum_allowable_length - data_len[i]) if (maximum_allowable_length - data_len[i])>0 else 0)+x_test_token[i] for i in range(len(x_test))])
print('padding done max_len: ', maximum_allowable_length, ' lenght: ',len(x_train_padded))
print(x_train_padded[0])
print(np.array(x_train_padded[0]))
print(x_train_padded.shape)
#print(y_train.shape)
print(y_train[:5])
model = Sequential()

embedding_size = 8
#Each token gets converted to a vector. This is trained along with the model
model.add(Embedding(input_dim = num_words,
                    output_dim = embedding_size,
                    input_length = maximum_allowable_length,
                    name='layer_1_embedding'))

model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(4, activation='relu', return_sequences=False))

model.add(Dense(1, activation='sigmoid'))#value will be between 0 and 1

#optimiser used is Adam

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

model.fit(x_train_padded, y_train, validation_split=0.5, epochs=10, batch_size=64)




