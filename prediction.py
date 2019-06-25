import tensorflow as tf
import scipy.io as sc
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn import preprocessing

def extract_features(samples, print_log=True):
    global prediction
    # this function is used to transfer one column label to one hot label
    def one_hot(y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]
		
	def __standardize(data) :
    # Store the data's original shape
    shape = data.shape
    # Flatten the data to 1 dimension
    print(data)
    # Find mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    print(mean)
    # Create a new array for storing standardized values
    standardized_values = list()
    # Iterate through every value in data
    for x in data:
        # standardize
        x_normalized = (x - mean) / std
        # Append it in the array
        standardized_values.append(x_normalized)
    # Convert to numpy array
    n_array = np.array(standardized_values)
    # Reshape the array to its original shape and return it.
    return np.reshape(n_array, shape)

    #  Data loading
    # insert here (don't remove the '1')
    for sample in samples:
        sample.append(1)
    all = np.array(samples)

    if print_log:
	    print('shape')
	    print (all.shape)

    np.random.shuffle(all)   # mix eeg_all
    # Get the 28000 samples of that subject
    final=1
    all=all[0:final]

    # Get the features
    feature_all =all[:,0:14]
    # Get the label
    label=all[:,14:15]

    # z-score

    if print_log:
	    print(feature_all)
	    print(feature_all.shape)
    no_fea=feature_all.shape[-1]
    label_all=label
    if print_log:
	    print("")
	    print (label_all)

    sns.set(font_scale=1.2)
    if print_log:
	    print("before")
	    print(feature_all)
    feature_all = (__standardize(feature_all))
    if print_log:
	    print("After")
	    print(feature_all)

    data = feature_all

    # Define sampling frequency and time vector
    sf = 160
    time = np.arange(data.shape[0]) / sf
    if print_log:
	    print(data.shape)
	    print('time')
	    print(time.shape)
    # Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(time, data, lw=1.5, color='k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.xlim([time.min(), time.max()])
    plt.title('EEG Data')
    sns.despine()

     # Define window length (4 seconds)
    win = 0.5 * sf
    freqs, psd = signal.welch(data, sf, nperseg=win)
    if print_log:
	    print(freqs)
	    print('psd')
	    print(psd.shape)

    n_classes=1
    ###CNN code,
    feature_all=feature_all# the input data of CNN
    if print_log:
	    print ("cnn input feature shape", feature_all.shape)
    n_fea=14
    if print_log:
	    print(n_fea)
    # label_all=one_hot(label_all)

    final=all.shape[0]
    middle_number=int(final*3/4)
    if print_log:
	    print("-----",middle_number)
    feature_training =feature_all[0:middle_number]
    feature_testing =feature_all[middle_number:final]
    label_training =label_all[0:middle_number]
    label_testing =label_all[middle_number:final]
    label_ww=label_all[middle_number:final]  # for the confusion matrix
    if print_log:
	    print ("label_testing",label_testing.shape)
    a=feature_training
    b=feature_testing
    if print_log:
	    print(feature_training.shape)
	    print(feature_testing.shape)

    keep=1
    batch_size=final-middle_number
    n_group=1
    train_fea=[]
    for i in range(n_group):
        f =a[(0+batch_size*i):(batch_size+batch_size*i)]
        train_fea.append(f)
    if print_log:
	    print("Here")
	    print (train_fea[0].shape)

    train_label=[]
    for i in range(n_group):
        f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
        train_label.append(f)
    if print_log:
	    print (train_label[0].shape)

    # the CNN code
    def compute_accuracy(v_xs, v_ys):

        y_pre = sess3.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess3.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep})
        return result

    #to creat a random weights
    def weight_variable(shape):
        # Outputs random values from a truncated normal distribution
        initial = tf.truncated_normal(shape, stddev=0.1)
        # A variable maintains state in the graph across calls to run().
        # You add a variable to the graph by constructing an instance of the class Variable.
        if print_log:
	        print('shape')
	        print(shape)
        return tf.Variable(initial)

    #random bias values
    def bias_variable(shape):
        # Creates a constant tensor
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        # the concolution layer x is the input
        # w is the weight and the stride is how many moves it makes in each dimention ie how many pixels
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # def max_pool_2x2(x):
    #     # stride [1, x_movement, y_movement, 1]
    #     return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
    #max pooling to reduce dimentionality .. here consider every 1*2 window
    def max_pool_1x2(x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, n_fea]) # 1*64
    ys = tf.placeholder(tf.float32, [None, n_classes])  # 2 is the classes of the data
    # Lookup what is keep_prob
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 1, n_fea, 1])
    if print_log:
	    print('x_image')
	    print(x_image)
	    print(x_image.shape)

    ## conv1 layer ##
    W_conv1 = weight_variable([1,1, 1,20]) # patch 1*1, in size is 1, out size is 2
    b_conv1 = bias_variable([20])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 1*64*2
    h_pool1 = max_pool_1x2(h_conv1)                          # output size 1*32x2

    ## conv2 layer ##
    # W_conv2 = weight_variable([1,1, 2, 4]) # patch 1*1, in size 2, out size 4
    # b_conv2 = bias_variable([4])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 1*32*4
    # h_pool2 = max_pool_1x2(h_conv2)                          # output size 1*16*4

    ## fc1 layer ## fc fully connected layer
    W_fc1 = weight_variable([1*int(n_fea/2)*20, 120])
    b_fc1 = bias_variable([120])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool1, [-1, 1*int(n_fea/2)*20])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([120, n_classes])
    b_fc2 = bias_variable([n_classes])
    # Multiplies matrix a by matrix b, producing a * b
    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Weight regulrization
    l2 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Getting the mean of the errors between the predication results and the class labels in the trainning data
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))+l2   # Softmax loss
    # Using optimizer
    train_step = tf.train.AdamOptimizer(0.04).minimize(cross_entropy)
    # Begin session to visit the nodes (tensors) of the graph
    sess3 = tf.Session()
    # Initializae all the defined variables
    init = tf.global_variables_initializer()
    # Visit the nodes of those variables
    sess3.run(init)
    # Total number of array elements which trigger summarization rather than full array
    #np.set_printoptions(threshold=np.nan)
    step = 1
    while step < 1500:
        # Train the model
        for i in range(n_group):
            sess3.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob:keep})
        # After 5 steps, use the model on the test data
        if step % 5 == 0:
            # Compute the cost using the cross entropy
            cost=sess3.run(cross_entropy, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
            # Compute the accuracy
            acc_cnn_t=compute_accuracy(b, label_testing)
            if print_log:
	            print('the step is:',step,',the acc is',acc_cnn_t,', the cost is', cost)
        step+=1
    acc_cnn=compute_accuracy(b, label_testing)
    feature_all_cnn=sess3.run(h_fc1_drop, feed_dict={xs: feature_all, keep_prob: keep})
    if print_log:
	    print ("the shape of cnn output features",feature_all.shape,label_all.shape)

    #######RNN
    tf.reset_default_graph()
    feature_all=feature_all
    no_fea=feature_all.shape[-1]
    if print_log:
	    print (no_fea)
    # The input to each LSTM layer must be a 3D
    # feature_all.reshape(samples-batch size-,time step, features)

    feature_all =feature_all.reshape([final,1,no_fea])
    #argmax returns the index with the largest value across axis of a tensor
    if print_log:
	    print (tf.argmax(label_all,1))
	    print (feature_all_cnn.shape)

    # middle_number=21000
    feature_training =feature_all
    feature_testing =feature_all
    label_training =label_all
    label_testing =label_all
    # print "label_testing",label_testing
    a=feature_training
    b=feature_testing
    if print_log:
	    print(feature_all)
	    print(feature_testing.shape)
    #264 dimention vector, that is passed to the next layer
    nodes=264
    #Used for Weight regulrization
    lameda=0.004
    #learning rate
    lr=0.005

    batch_size=final-middle_number
    train_fea=[]
    n_group=1
    for i in range(n_group):
        f =a[(0+batch_size*i):(batch_size+batch_size*i)]
        train_fea.append(f)

    if print_log:
	    print("here0")
	    print (train_fea[0].shape)

    train_label=[]
    for i in range(n_group):
        f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
        train_label.append(f)
    if print_log:
	    print (train_label[0].shape)


    # hyperparameters

    n_inputs = no_fea
    n_steps = 1 # time steps
    n_hidden1_units = nodes   # neurons in hidden layer
    n_hidden2_units = nodes
    n_hidden3_units = nodes
    n_hidden4_units=nodes
    n_classes = n_classes

    # tf Graph input

    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Define weights
    #tf.random_normal: Outputs random values from a normal distribution
    weights = {

    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
    'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),

    'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
    'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
    'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),

    'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
    }

    biases = {
    #tf.constant result a 1-D tensor of value 0.1
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),

    'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ])),
    'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
    'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),

    'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True)
    }


    def RNN(X, weights, biases):
        # hidden layer for input to cell
        ########################################

        # transpose the inputs shape from
        X = tf.reshape(X, [-1, n_inputs])

        # into hidden
        #there are n input and output we take only the last output to feed to the next layer
        X_hidd1 = tf.matmul(X, weights['in']) + biases['in']
        X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
        X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
        X_hidd4 = tf.matmul(X_hidd3, weights['hidd4']) + biases['hidd4']
        X_in = tf.reshape(X_hidd4, [-1, n_steps, n_hidden4_units])


        # cell
        ##########################################

        # basic LSTM Cell.
        # 1-layer LSTM with n_hidden units.
        # creates a LSTM layer and instantiates variables for all gates.
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
        # 2nd layer LSTM with n_hidden units.
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
        # Adding an additional layer to inprove the accuracy
        # RNN cell composed sequentially of multiple simple cells.

        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # lstm cell is divided into two parts (c_state, h_state)
        #Initializing the zero state
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        with tf.variable_scope('lstm1', reuse=tf.AUTO_REUSE):
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

        # hidden layer for output as the final results
        #############################################
        if print_log:
	        print("before")
	        print(outputs)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
        if print_log:
	        print("after")
	        print(outputs)
        #there are n input and n output we take only the last output to feed to the next layer
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return results, outputs[-1]

    #################################################################################################################################################
    pred,Feature = RNN(x, weights, biases)
    lamena =lameda
    l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2  # Softmax loss
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        # train_op = tf.train.AdagradOptimizer(l).minimize(cost)
        # train_op = tf.train.RMSPropOptimizer(0.00001).minimize(cost)
        # train_op = tf.train.AdagradDAOptimizer(0.01).minimize(cost)
        # train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
    # pred_result =tf.argmax(pred, 1)
    label_true =tf.argmax(y, 1)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    confusion_m=tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(pred, 1))
    #starting sessions
    with tf.Session() as sess:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        step = 0
        if print_log:
	        print(train_fea[0])
	        print(train_label[0])

        #downloaded = drive.CreateFile({'id':'10p_NuiBV2Or2sk6cm0yPLfu9tJ2lXEKg'})
        #f2 = downloaded.GetContentString()

        #filename = "/home/xiangzhang/scratch/results/rnn_acc.csv"
        #f2 = open(filename, 'wb')
        while step < 2500:
                sess.run(train_op, feed_dict={
                    x: train_fea[0],
                    y: train_label[0],
                })
                if sess.run(accuracy, feed_dict={x: b,y: label_testing,})>0.96:
                    print(
                    "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
                    sess.run(accuracy, feed_dict={
                        x: b,
                        y: label_testing,
                    }))

                    break
                if step % 5 == 0:
                    hh=sess.run(accuracy, feed_dict={
                        x: b,
                        y: label_testing,
                    })
                    #f2.write(str(hh)+'\n')
                    print(", The step is:",step,", The accuracy is:", hh, "The cost is :",sess.run(cost, feed_dict={
                        x: b,
                        y: label_testing,
                    }))
                step += 1

        ##confusion matrix
        feature_0=sess.run(Feature, feed_dict={x: train_fea[0]})
        for i in range(1,n_group):
            feature_11=sess.run(Feature, feed_dict={x: train_fea[i]})
            feature_0=np.vstack((feature_0,feature_11))

        if print_log:
	        print (feature_0.shape)
        feature_b = sess.run(Feature, feed_dict={x: b})
        feature_all_rnn=np.vstack((feature_0,feature_b))

        confusion_m=sess.run(confusion_m, feed_dict={
                    x: b,
                    y: label_testing,
                })
        if print_log:
	        print (confusion_m)
        ## predict probility
        # pred_prob=sess.run(pred, feed_dict={
        #             x: b,
        #             y: label_testing,
        #         })
        # # print pred_prob


        #print ("RNN train time:", time4 - time3, "Rnn test time", time5 - time4, 'RNN total time', time5 - time3)

        ##AE
    if print_log:
	    print (feature_all_rnn.shape, feature_all_cnn.shape)
    new_feature_all_rnn = feature_all_rnn[0:1, :]
    if print_log:
	    print(new_feature_all_rnn.shape)
    # stacks the featurese from RNN and CNN in a horizontal stack
    feature_all=np.hstack((new_feature_all_rnn,psd) )
    feature_all=np.hstack((feature_all,feature_all_cnn))
    if print_log:
	    print(psd.shape, feature_all.shape)
    no_fea=feature_all.shape[-1]

    # feature_all =feature_all.reshape([28000,1,no_fea])
    if print_log:
	    print("all features")
	    print(feature_all.shape)
    # middle_number=21000
    feature_training =feature_all
    feature_testing =feature_all
    label_training =label_all
    label_testing =label_all
    # print "label_testing",label_testing
    a=feature_training
    b=feature_testing
    feature_all=feature_all

    train_fea=feature_all

    #dividing the input into three groups
    group=1
    display_step = 10
    #An epoch is a full iteration over samples!!!! training cycle
    training_epochs = 400

    # Network Parameters
    n_hidden_1 = 800 # 1st layer num features, should be times of 8


    n_hidden_2=100

    n_input_ae = no_fea # MNIST data input (img shape: 28*28)
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input_ae])
    if print_log:
	    print("X")
	    print(X)

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input_ae, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), #NOT USED !!!
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input_ae])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input_ae])),
    }


    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        #Sigmoid function outputs in the range (0, 1), it makes it ideal for binary classification problems
        #there are n input and output we take only the last output to feed to the next layer
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        return layer_1


    # Building the decoder
    def decoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        #Sigmoid function outputs in the range (0, 1), it makes it ideal for binary classification problems
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_1

    for ll in range(1):
        learning_rate = 0.2
        for ee in range(1):
            # Construct model
            encoder_op = encoder(X)
            if print_log:
	            print("Encoder")
	            print(encoder_op)
            decoder_op = decoder(encoder_op)
            # Prediction
            y_pred = decoder_op
            # Targets (Labels) are the input data, as the auto encoder tries to make output as similar as possible to the input.
            y_true = X

            # Define loss and optimizer, minimize the squared error
            cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            # cost = tf.reduce_mean(tf.pow(y_true, y_pred))
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

            # Initializing the variables
            init = tf.global_variables_initializer()

            # Launch the graph
            # saves and restore variables
            saver = tf.train.Saver()
            with tf.Session() as sess1:
                sess1.run(init)
                saver = tf.train.Saver()
                # Training cycle
                for epoch in range(training_epochs):
                    # Loop over all batches
                    for i in range(group):
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess1.run([optimizer, cost], feed_dict={X: a})
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        if print_log:
	                        print("Epoch:", '%04d' % (epoch+1),
                              "cost=", "{:.9f}".format(c))
                if print_log:
	                print("Optimization Finished!")
                a = sess1.run(encoder_op, feed_dict={X: a})
                b = sess1.run(encoder_op, feed_dict={X: b})
    return a, label_testing

if __name__ == "__main__":
    samples_test = [
    [-73.0, 3.0, 274.0, -23.0, -25.0, -9.0, -24.0, 154.0, -25.0, -61.0, -578.0, -723.0, -357.0, 237.0],
    [358.0, 23.0, -662.0, 72.0, 61.0, 29.0, 19.0, -37.0, 23.0, 15.0, 102.0, -164.0, 151.0, 98.0],
    [-35.0, 1.0, 148.0, -3.0, -2.0, 4.0, 10.0, 214.0, 294.0, 425.0, 726.0, 581.0, 516.0, -88.0],
    [77.0, -1.0, -490.0, 15.0, 14.0, 6.0, 17.0, -948.0, -865.0, -825.0, -874.0, -941.0, -854.0, -765.0],
    [150.0, 19.0, -10.0, 41.0, 44.0, 25.0, -0.0, 414.0, 567.0, 456.0, 183.0, 125.0, 560.0, 361.0],
    [-94.0, -34.0, -398.0, -25.0, -28.0, -19.0, 22.0, -627.0, -1023.0, -979.0, -951.0, -760.0, -1007.0, -784.0],
    [-216.0, -39.0, -361.0, -61.0, -78.0, -34.0, -8.0, -1365.0, -1819.0, -1862.0, -1589.0, -1342.0, -1706.0, -1163.0],
    [21.0, -15.0, -294.0, -4.0, -12.0, -12.0, -4.0, 233.0, -111.0, -144.0, 14.0, 117.0, -28.0, -11.0],
    [-64.0, -10.0, -410.0, -30.0, -48.0, -16.0, -35.0, -467.0, -1281.0, -1326.0, -1139.0, -959.0, -1141.0, -399.0],
    [31.0, 5.0, -46.0, 3.0, 1.0, -1.0, -17.0, 185.0, 200.0, 160.0, 233.0, 220.0, 234.0, 143.0]
    ]
    a, label_testing = extract_features(samples_test, print_log=True)
    print(a)
