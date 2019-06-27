import tensorflow as tf
import scipy.io as sc
import numpy as np
import random
from scipy.fftpack import rfft
import time
from sklearn import preprocessing

def predict_cnn(samples):
    # this function is used to transfer one column label to one hot label
    def one_hot(y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        print("one hot")
        print(y_.shape)
        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]

    #  Data loading
    feature = sc.loadmat("recordings/louay2_no_eyes.mat")

    all = feature['Eddeny3a2lk']
    print('Feature')
    print(feature)

    print('shape')
    print (all.shape)
    print('shape')
    print (all.shape)
    np.random.shuffle(all)   # mix eeg_all
    # Get the 28000 samples of that subject
    final=len(all)
    print(final)
    all=all[0:final]
    temp = all[final-1]
    temp = np.reshape(temp, (1,15))
    print(temp.shape)
    all = np.append(all, temp, axis=0)

    # Get the features
    feature_all =all[:,0:14]


    # Get the label
    labels=all[:,14]

    # z-score

    print("Feature All")
    print(feature_all)
    # transposed = [list(i) for i in zip(*feature_all)] ## Transpose so each sub-array is the data of a channel in time domain
    # fft_transposed = [rfft(np.array(i)) for i in transposed] ## Compute FFT (real values)
    # fft_samples = [list(np.round(i, 0)) for i in zip(*fft_transposed)] ## Transpose again so each channel is in a column instead of a row, rounds to the nearest unit while transposing
    # feature_all = fft_samples[0:]

    # feature_all = np.asarray(feature_all)
    no_fea=feature_all.shape[-1]
    labels = labels.astype(int)

    print("labels")
    print(labels)
    labels_all=one_hot(labels)
    print("")
    print (labels_all)
    feature_all = preprocessing.scale(feature_all)

    n_classes=4
    ###CNN code,
    print ("cnn input feature shape", feature_all.shape)
    n_fea=feature_all.shape[-1]
    print(n_fea)
    # labels_all=one_hot(labels_all)

    final=all.shape[0]
    middle_number=int(final*3/4)
    print("-----",middle_number)
    feature_training =feature_all[0:middle_number]
    feature_testing =feature_all[middle_number:final]
    label_training =labels_all[0:middle_number]
    label_testing =labels_all[middle_number:final]
    label_ww=labels_all[middle_number:final]  # for the confusion matrix
    print ("label_testing",label_testing.shape)
    # a = np.append(feature_training, feature_testing, axis=0)
    a = np.append(feature_training, feature_testing, axis=0)
    label_training = labels_all
    print(np.array(samples).shape)
    print(feature_testing.shape)
    # b = np.append(np.array(samples), feature_testing, axis=0)
    b = np.array(samples)
    print(feature_training.shape)
    print(feature_testing.shape)

    print("Input shape")
    print(b.shape)
    print("Label shape")
    print(label_testing.shape)

    keep=1
    batch_size=final-middle_number
    n_group=3
    train_fea=[]
    for i in range(n_group):
        f =a[(0+batch_size*i):(batch_size+batch_size*i)]
        train_fea.append(f)
    print("Here")
    print (train_fea[0].shape)

    train_label=[]
    for i in range(n_group):
        f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
        train_label.append(f)
    print (train_label[0].shape)

    # the CNN code
    def compute_accuracy(v_xs, v_ys):
        global prediction
        y_pre = sess3.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        print("prediction")
        print(np.argmax(y_pre,1))
        print("origin")
        print(np.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess3.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep})
        return result

    #to creat a random weights
    def weight_variable(shape):
        # Outputs random values from a truncated normal distribution
        initial = tf.truncated_normal(shape, stddev=0.1)
        # A variable maintains state in the graph across calls to run().
        # You add a variable to the graph by constructing an instance of the class Variable.
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

    def predict_data(v_xs):
        y_pre = sess3.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
        print("prediction")
        return np.argmax(y_pre,1)

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, n_fea]) # 1*14
    ys = tf.placeholder(tf.float32, [None, n_classes])  # 2 is the classes of the data
    # Lookup what is keep_prob
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 1, n_fea, 1])
    print('xs')
    print(xs)
    print(xs.shape)
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
    while step < 300:
        # Train the model
        print("Step is " + str(step))
        for i in range(n_group):
            sess3.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob:keep})
        # After 5 steps, use the model on the test data
        step+=1
    prediction_arg_max = predict_data(b)
    feature_all_cnn=sess3.run(h_fc1_drop, feed_dict={xs: feature_all, keep_prob: keep})
    print ("the shape of cnn output features",feature_all.shape,labels_all.shape)
    return prediction_arg_max[0 : len(samples)]
