
    # Gradient Descent
    rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name="Rewards")
    actions = tf.placeholder(shape=[None, ], dtype=tf.float32, name="Actions")
    #print(actions[-1].shape)
    # print(n_classes)
    #one_hot = tf.one_hot(actions, tf.cast(n_classes, tf.int32), name="One_Hot")
    cross_entropies = tf.losses.softmax_cross_entropy(
        labels=actions,
        logits= a4)
    loss = tf.reduce_sum( rewards * cross_entropies)
    # Training Operation 
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=.99)
    train_op =  optimizer.minimize(loss)

