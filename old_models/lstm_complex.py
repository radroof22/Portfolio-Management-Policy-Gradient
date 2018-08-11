X = tf.placeholder(dtype=tf.float32, shape=[None, n_days, n_features], name="Input_Data")
rnn_cell = tf.contrib.rnn.BasicLSTMCell( 256, reuse=tf.AUTO_REUSE, name="LSTM_Cell")
r_outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)
a1 = tf.layers.dense(r_outputs, 128, activation=tf.nn.relu, name='Activation_1',  reuse=tf.AUTO_REUSE)
dropout1 = tf.layers.dropout(a1, rate=.5, training=True, name="Dropout_1")
a2 = tf.layers.dense(dropout1, 64, activation=tf.nn.relu, name="Activation_2",  reuse=tf.AUTO_REUSE)
dropout2 = tf.layers.dropout(a2, rate=.5, training=True, name="Dropout_2")
a3 = tf.layers.dense(dropout2, 32, activation=tf.nn.relu, name="A3",  reuse=tf.AUTO_REUSE)
a4 = tf.layers.dense(a3, n_actions, name="A4",  reuse=tf.AUTO_REUSE)[:,1]
outputs = tf.nn.softmax(a4, name="Outputs")
choice = tf.argmax(outputs, axis=1, name="Choice")[0]

## Training Procedur
rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="Rewards")
actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")

# One hot array of actions
one_hot_actions = tf.one_hot(actions, n_actions)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=a4, labels=one_hot_actions)

loss = tf.reduce_mean(cross_entropy * rewards, name="Loss")

# Create op to update gradients
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#update_gradients = optimizer.apply_gradients(zip(gradients_to_apply, tf.trainable_variables()))
