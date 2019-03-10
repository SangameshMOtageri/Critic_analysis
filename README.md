# Critic_analysis
Model trained to identify whether a review is a positive one or a negative one.
Model is an LSTM based RNN implemented using Keras, tensorflow libraries.
Results:
epochs_count:2
num_words:600
3ms/step - loss: 0.6897 - acc: 0.5765 - val_loss: 0.6790 - val_acc: 0.6445
epochs_count:5
num_words:600
3ms/step - loss: 0.6793 - acc: 0.5964 - val_loss: 0.6650 - val_acc: 0.6052
epochs_count:5
num_words:1000
3ms/step - loss: 0.5911 - acc: 0.7207 - val_loss: 0.6497 - val_acc: 0.6769
epochs_count:10
num_words:2000
3ms/step - loss: 0.3154 - acc: 0.8537 - val_loss: 1.0960 - val_acc: 0.7149
