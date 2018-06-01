import time
import numpy as np
import argparse
import tensorflow as tf
import matplotlib
import math
import os
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_LABELS = 47
rnd = np.random.RandomState(123)
tf.set_random_seed(123)

# CNN_MODEL_PATH = "./models/cnn_model/cnn"
# AE_MODEL_PATH = "./models/ae_model/ae"

cnn_saver = None
sess = tf.Session()
params, train_op, loss, out, enc_conv3, accuracy = None, None, None, None, None, None

# Following functions are helper functions that you can feel free to change
def convert_image_data_to_float(image_raw):
    img_float = tf.expand_dims(tf.cast(image_raw, tf.float32) / 255, axis=-1)
    return img_float

def plot_graph(filename, data, xlabel, ylabel, title):
    if not os.path.exists("./graph"):
        os.makedirs("./graph")

    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(np.array([int(x) for x in range(len(data))]), data,'.-')
    plt.show()
    plt.savefig("./graph/" + filename)

def visualize_ae(i, iter, x, features, reconstructed_image):
    '''
    This might be helpful for visualizing your autoencoder outputs
    :param i: index
    :param x: original data
    :param features: feature maps
    :param reconstructed_image: autoencoder output
    :return:
    '''
    plt.figure()
    plt.imshow(x[i, :, :], cmap="gray")
    plt.savefig("./fig/fig-input-" + str(i) + "_" + str(iter) + ".pdf")
    plt.figure()
    plt.imshow(reconstructed_image[i, :, :, 0], cmap="gray")
    plt.savefig("./fig/fig-reconstructed-" + str(i) + "_" + str(iter) + ".pdf")
    plt.figure()
    plt.imshow(np.reshape(features[i, :, :, :], (7, -1), order="F"), cmap="gray",)
    plt.savefig("./fig/fig-features-" + str(i) + "_" + str(iter) + ".pdf")

def cnn_model(x, y, lr, mm, mode):
    with tf.variable_scope("cnn-" + mode) as scope:
        img_float = convert_image_data_to_float(x)

        # 4 convolutional layers
        conv1 = tf.layers.conv2d(
            inputs=img_float,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            name='conv1',kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            strides=2,
            name='conv2',kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            name='conv3',kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            strides=2,
            name='conv4',kernel_initializer=tf.contrib.layers.xavier_initializer())

        # flatten
        img_flattened = tf.contrib.layers.flatten(conv4)

        # fully connected layer 1024 hidden size with relu
        fc_weight = tf.get_variable("fc_weight", shape=(img_flattened.shape[1],1024), initializer=tf.contrib.layers.xavier_initializer())
        fc_biases = tf.Variable(tf.zeros([1024]))
        fc = tf.nn.relu(tf.matmul(img_flattened, fc_weight) + fc_biases)

        # softmax
        output_weight = tf.get_variable("weight", shape=(1024, NUM_LABELS), initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(fc, output_weight)
        softmax_loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        loss = tf.identity(softmax_loss, name="loss")
        # calculating the accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.cast(y,tf.int64)),dtype=tf.float32), name="accuracy")

        # update the weights
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)

        accumulations = []
        new_grads = []
        new_params = []
        i = 0
        for id in range(len(grads)):
            grad = grads[i]
            param = params[i]
            if grad is not None:
                accumulations.append(tf.Variable(np.zeros(grad.shape, dtype=np.float32)))
                new_grads.append(grad)
                new_params.append(param)
            i+=1
        grads = new_grads
        params = new_params

        # accumulations = [tf.Variable(np.zeros(grad.shape, dtype=np.float32)) for grad in grads]
        var_updates = []
        for grad, accumulation, var in zip(grads, accumulations, params):
            var_updates.append(tf.assign(accumulation, mm * accumulation + grad))
            var_updates.append(tf.assign_add(var, -lr*accumulation))
        train_op = tf.group(*var_updates)

        """
        Formula:
        accumulation = momentum * accumulation + grad
        var -= lr * accumulation
        """
        # train_op = tf.train.MomentumOptimizer(momentum=mm, learning_rate=lr).minimize(loss)
    
        return params, train_op, loss, accuracy

def ae_model(x, lr, mm, mode):
    with tf.variable_scope("aecnn-" + mode) as scope:
        img_float = convert_image_data_to_float(x)

        ##########################
        ######## ENCODER #########
        ##########################

        # 3 convolutional layers
        enc_conv1 = tf.layers.conv2d(
            inputs=img_float,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            name='enc_conv1',kernel_initializer=tf.contrib.layers.xavier_initializer())

        enc_conv2 = tf.layers.conv2d(
            inputs=enc_conv1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            strides=2,
            name='enc_conv2',kernel_initializer=tf.contrib.layers.xavier_initializer())

        enc_conv3 = tf.layers.conv2d(
            inputs=enc_conv2,
            filters=2,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            name='enc_conv3',kernel_initializer=tf.contrib.layers.xavier_initializer())

        output_conv = tf.identity(enc_conv3)

        ##########################
        ######## DECODER #########
        ##########################

        dec_conv1 = tf.layers.conv2d_transpose(
            inputs=enc_conv3,
            filters=2,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            name='dec_conv1',kernel_initializer=tf.contrib.layers.xavier_initializer())

        dec_conv2 = tf.layers.conv2d_transpose(
            inputs=dec_conv1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            strides=2,
            name='dec_conv2',kernel_initializer=tf.contrib.layers.xavier_initializer())

        dec_conv3 = tf.layers.conv2d_transpose(
            inputs=dec_conv2,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            name='dec_conv3',kernel_initializer=tf.contrib.layers.xavier_initializer())

        # output layer
        out = tf.identity(dec_conv3)

        ##########################
        ####### EVALUATION #######
        ##########################

        params = tf.trainable_variables()
        loss = 0
        mse_loss = 0 
        regularizer = 0

        learning_rate = lr
        
        for p in params:
            regularizer += 0.01 * tf.reduce_mean(tf.square(p))
        mse_loss += tf.div(tf.reduce_mean(tf.square(tf.subtract(out, img_float))),2,name="mse")
        loss = mse_loss + regularizer

        # update the weights
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        accumulations = [tf.Variable(np.zeros(grad.shape, dtype=np.float32)) for grad in grads]
        var_updates = []
        for grad, accumulation, var in zip(grads, accumulations, params):
            var_updates.append(tf.assign(accumulation, mm * accumulation + grad))
            var_updates.append(tf.assign_add(var, -learning_rate*accumulation))
        train_op = tf.group(*var_updates)

        return params, train_op, loss, out, output_conv

# Major interfaces
def train_cnn(x, y, placeholder_x, placeholder_y, batch_size, momentum, lr):
    print("TRAIN CNN")
    start_time = time.time()
    num_iterations = 20
    params, train_op, loss, accuracy = cnn_model(placeholder_x, placeholder_y, lr, momentum, "train")
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    cnn_saver = tf.train.Saver()

    n_sample = x.shape[0]
    indices = [x for x in range(n_sample)]
    np.random.shuffle(indices)
    new_x = x[indices]
    new_y = y[indices]

    n_train_sample = n_sample

    train_data = new_x[:n_train_sample]
    train_label = new_y[:n_train_sample]

    print("total sample:", n_sample,"train sample:", n_train_sample)

    n_train_batch = math.ceil(n_train_sample/batch_size)

    losses = []
    accus = []
    for n_iter in range(num_iterations):
        total_train_loss = 0
        total_train_accu = 0
    
        # TRAIN
        pbar = tqdm(range(n_train_batch), total=n_train_batch)
        for i in pbar:
            batch_input = train_data[i * batch_size:(i+1) * batch_size]
            batch_label = train_label[i * batch_size:(i+1) * batch_size]

            feed_dict = {placeholder_x: batch_input, placeholder_y: batch_label}
            _, l, accu  = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
            input_batch_size = batch_input.shape[0]

            total_train_loss += l
            total_train_accu += accu * input_batch_size
            
            pbar.set_description('L:{:.5f}, A:{:.5f}'.format(total_train_loss / (i+1), total_train_accu / n_train_sample))
        losses.append(total_train_loss / (i+1))
        accus.append(total_train_accu / n_train_sample)

        # n_test_sample = len(test_x)

        # print("test sample:", n_test_sample)

        # n_test_batch = math.ceil(n_test_sample/batch_size)
        # total_test_loss = 0
        # result_accuracy = 0

        # # TEST
        # print("TEST")
        # pbar = tqdm(range(n_test_batch), total=n_test_batch)
        # for i in pbar:
        #     batch_input = x[i * batch_size:(i+1) * batch_size]
        #     batch_label = y[i * batch_size:(i+1) * batch_size]

        #     feed_dict = {placeholder_x: batch_input, placeholder_y: batch_label}
        #     l, accu  = sess.run([loss, accuracy], feed_dict=feed_dict)
        #     total_test_loss += l
        #     input_batch_size = batch_input.shape[0]
        #     result_accuracy += accu * input_batch_size

        #     pbar.set_description('L:{:.2f}, ACCU:{:.2f}'.format(total_test_loss / (i+1), result_accuracy / n_test_sample))

    # Save the variables to disk.
    # save_path = cnn_saver.save(sess, CNN_MODEL_PATH)
    # print("Model saved in file: %s" % save_path)

    plot_graph("cnn_train_losses_" + ".pdf", losses, "iteration", "train loses", "Train losses")
    plot_graph("cnn_train_accus_" + ".pdf", accus, "iteration", "train accus", "Train accus")       
    print("train elapsed time:", time.time() - start_time)              

    return params, train_op, loss, accuracy

def cross_valid_cnn(x, y, placeholder_x, placeholder_y):
    print("CROSS VALIDATION - GRID SEARCH CNN")
    num_iterations = 20

    n_sample = x.shape[0]
    indices = [x for x in range(n_sample)]
    np.random.shuffle(indices)
    new_x = x[indices]
    new_y = y[indices]

    best_val_accu = 0      
    best_batch_size = -1
    best_lr = -1
    best_momentum = -1

    et = 0

    # grid search
    bsz = [32, 64]
    lrs = [0.1, 0.01]
    momentum = [0.2, 0.6]
    
    elapsed_times = {}
    all_train_losses = {}
    all_train_accus = {}
    all_val_losses = {}
    all_val_accus = {}
    modes = []

    for lr in lrs:
        for batch_size in bsz:
            for mm in momentum:
                mode = "lr=" + str(lr) + ", bsz=" + str(bsz) + ", mm=" + str(mm)

                best_config_val_accu = 0
                start_time = time.time()

                train_losses = []
                train_accus = []
                val_losses = []
                val_accus = []

                mode = "val_lr_" + str(lr) + "_mm_" + str(mm) + "_bsz_" + str(batch_size)
                params, train_op, loss, accuracy = cnn_model(placeholder_x, placeholder_y, lr, mm, mode)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
            
                    n_train_sample = math.ceil(0.8 * n_sample)
                    n_valid_sample = n_sample - n_train_sample

                    train_data = new_x[:n_train_sample]
                    train_label = new_y[:n_train_sample]
                    valid_data = new_x[n_train_sample:]
                    valid_label = new_y[n_train_sample:]

                    print("total sample:", n_sample,"train sample:", n_train_sample, "valid sample:", n_valid_sample)

                    n_train_batch = math.ceil(n_train_sample/batch_size)
                    n_val_batch = math.ceil(n_valid_sample/batch_size)

                    for n_iter in range(num_iterations):
                        print("\nEPOCH:", str(n_iter+1), "bsz:", str(batch_size),"mm:", str(mm), "lr:", str(lr))
                        total_train_loss = 0
                        total_train_accu = 0
                        total_val_loss = 0
                        total_val_accu = 0

                        # TRAIN
                        print("TRAIN")
                        pbar = tqdm(range(n_train_batch), total=n_train_batch)
                        for i in pbar:
                            batch_input = train_data[i * batch_size:(i+1) * batch_size]
                            batch_label = train_label[i * batch_size:(i+1) * batch_size]

                            feed_dict = {placeholder_x: batch_input, placeholder_y: batch_label}
                            _, l, accu  = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
                            total_train_loss += l
                            total_train_accu += accu
                            input_batch_size = batch_input.shape[0]

                            pbar.set_description('L:{:.2f}'.format(total_train_loss / (i+ 1)))

                        train_loss = total_train_loss / n_train_batch
                        train_accu = total_train_accu / n_train_batch
                        print("train accu", str(train_accu))

                        # VALID
                        print("VALID")
                        pbar = tqdm(range(n_val_batch), total=n_val_batch)
                        for i in pbar:
                            batch_input = valid_data[i * batch_size:(i+1) * batch_size]
                            batch_label = valid_label[i * batch_size:(i+1) * batch_size]

                            feed_dict = {placeholder_x: batch_input, placeholder_y: batch_label}
                            l, accu  = sess.run([loss, accuracy], feed_dict=feed_dict)
                            total_val_loss += l
                            total_val_accu += accu
                            input_batch_size = batch_input.shape[0]

                            pbar.set_description('L:{:.2f}, ACCU:{:.2f}'.format(total_val_loss / (i+ 1), total_val_accu / (i+1)))

                        val_loss = total_val_loss / n_val_batch
                        val_accu = total_val_accu / n_val_batch
                        print("val loss", str(val_loss), "val accu", str(val_accu))

                        train_accus.append(train_accu)
                        train_losses.append(train_loss)
                        val_accus.append(val_accu)
                        val_losses.append(val_loss)

                        if best_config_val_accu < val_accu:
                            et = 0
                            best_config_val_accu = val_accu

                            if best_val_accu < best_config_val_accu:
                                best_val_accu = best_config_val_accu
                                best_batch_size = batch_size
                                best_lr = lr
                                best_momentum = mm
                        else:
                            et += 1

                        if et == 2:
                            print("EARLY STOPPING best config: accu=" + str(best_config_val_accu) + " lr=" + str(lr) + " bsz=" + str(batch_size) + " n_iter=" + str(n_iter+1))
                            break
                
                end_time = time.time()
                elapsed_time = start_time - end_time
                elapsed_times[mode] = elapsed_time
                all_train_losses[mode] = train_losses
                all_train_accus[mode] = train_accus
                all_val_losses[mode] = val_losses
                all_val_accus[mode] = val_accus
                modes.append(mode)

                plot_graph("cnn_cross_val_train_losses_" + mode + ".pdf", train_losses, "iteration", "train loses", "Train losses " + mode)
                plot_graph("cnn_cross_val_train_accus_" + mode + ".pdf", train_accus, "iteration", "train accus", "Train accus " + mode)
                plot_graph("cnn_cross_val_val_losses_" + mode + ".pdf", val_losses, "iteration", "val losses", "Val losses " + mode)
                plot_graph("cnn_cross_val_val_accus_" + mode + ".pdf", val_accus, "iteration", "val accus", "Val accus " + mode)
                print("mode:", mode, "elapsed time:", elapsed_time,"sec")

    for i in range(len(mode)):
        # print(mode, "train loss:", all_train_losses[mode])
        # print(mode, "train accu:", all_train_accus[mode])
        # print(mode, "val loss:", all_val_losses[mode])
        # print(mode, "val accu:", all_val_accus[mode])
        print(mode, "elapsed time:", elapsed_times[mode])

    print("BEST CONFIG FROM GRID SEARCH:")
    print("accu:", best_val_accu, "batch_size:", best_batch_size, "lr:", best_lr, "momentum:", best_momentum)

def test_cnn(x, y, placeholder_x, placeholder_y, batch_size, momentum, lr, params, train_op, loss, accuracy):
    print("TEST CNN")

    # params, train_op, loss, accuracy = cnn_model(placeholder_x, placeholder_y, lr, momentum, "test")

    # sess.run(tf.global_variables_initializer())
    # cnn_saver = tf.train.import_meta_graph('./models/cnn_model/cnn.meta')
    # sess.run(tf.local_variables_initializer())
    
    # Restore variables from disk.
    # cnn_saver = tf.train.Saver()
    # cnn_saver.restore(sess, CNN_MODEL_PATH)
    # graph = tf.get_default_graph()
    # print("Model restored.")

    n_test_sample = len(x)

    print("test sample:", n_test_sample)

    n_test_batch = math.ceil(n_test_sample/batch_size)
    total_test_loss = 0
    result_accuracy = 0

    # TEST
    print("TEST")
    pbar = tqdm(range(n_test_batch), total=n_test_batch)
    for i in pbar:
        batch_input = x[i * batch_size:(i+1) * batch_size]
        batch_label = y[i * batch_size:(i+1) * batch_size]

        feed_dict = {placeholder_x: batch_input, placeholder_y: batch_label}
        l, accu  = sess.run([loss, accuracy], feed_dict=feed_dict)
        total_test_loss += l
        input_batch_size = batch_input.shape[0]
        result_accuracy += accu * input_batch_size

        pbar.set_description('L:{:.2f}, ACCU:{:.2f}'.format(total_test_loss / (i+1), result_accuracy / n_test_sample))

    return result_accuracy / n_test_sample

def train_ae(x, placeholder_x,  batch_size, lr, mm):
    print("TRAIN AE")
    num_iterations = 20
    params, train_op, loss, out, enc_conv3 = ae_model(placeholder_x, lr, mm, "train-ae"+str(lr))

    ae_saver = tf.train.Saver()
    losses = []
    # with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    n_sample = x.shape[0]
    indices = [x for x in range(n_sample)]
    np.random.shuffle(indices)
    new_x = x[indices]

    n_train_sample = n_sample

    train_data = new_x[:n_train_sample]

    print("total sample:", n_sample,"train sample:", n_train_sample)

    n_train_batch = math.ceil(n_train_sample/batch_size)
    start_time = time.time()
    for n_iter in range(num_iterations):
        total_train_loss = 0
    
        # TRAIN
        print("TRAIN")
        pbar = tqdm(range(n_train_batch), total=n_train_batch)
        for i in pbar:
            batch_input = train_data[i * batch_size:(i+1) * batch_size]
            feed_dict = {placeholder_x: batch_input}
            # _, l, enc_conv3, predict  = sess.run([train_op, loss, enc_conv3, out], feed_dict=feed_dict)
            _, l, outenc_conv3, predict  = sess.run([train_op, loss, enc_conv3, out], feed_dict=feed_dict)
            total_train_loss += l
            input_batch_size = batch_input.shape[0]
            pbar.set_description('L:{:.5f}'.format(total_train_loss / (i+1)))

        losses.append(total_train_loss / (i+1))

        if n_iter == num_iterations-1:
            for j in range(5):
                visualize_ae(j, n_iter, np.squeeze(batch_input), np.squeeze(outenc_conv3), np.squeeze(predict))

    plot_graph("ae_losses.pdf", losses, "iteration", "train losses", "Train losses ")
    print("train time:", time.time()-start_time)

    # print("TEST")
    # n_test_batch = math.ceil(len(x_test) / batch_size)
    # total_test_loss = 0

    # pbar = tqdm(range(n_test_batch), total=n_test_batch)
    # for i in pbar:
    #     batch_input = x_test[i * batch_size:(i+1) * batch_size]

    #     feed_dict = {placeholder_x: batch_input}
    #     l = sess.run([loss], feed_dict=feed_dict)
    #     total_test_loss += l[0]

    #     pbar.set_description('L:{:.2f}'.format(total_test_loss / (i+1)))

    # # Save the variables to disk.
    # save_path = ae_saver.save(sess, AE_MODEL_PATH)
    # print("Model saved in file: %s" % save_path)

    return params, train_op, loss, out, enc_conv3 

def cross_valid_ae(x, placeholder_x):
    print("CROSS VALIDATION - GRID SEARCH AE")
    num_iterations = 30

    n_sample = x.shape[0]
    indices = [x for x in range(n_sample)]
    np.random.shuffle(indices)
    new_x = x[indices]

    best_val_loss= 10000000000      
    best_batch_size = -1
    best_lr = -1
    best_momentum = -1

    et = 0

    # grid search
    bsz = [256, 512]
    lrs = [0.1, 0.2]
    momentum = [0.2, 0.6]

    elapsed_times = {}
    all_train_losses = {}
    all_val_losses = {}
    modes = []
    
    for lr in lrs:
        for batch_size in bsz:
            for mm in momentum:
                best_config_val_loss = 1000000

                start_time = time.time()
                train_losses = []
                val_losses = []

                mode = "val_lr_" + str(lr) + "_mm_" + str(mm) + "_bsz_" + str(batch_size)
                params, train_op, loss, out, enc_conv3  = ae_model(placeholder_x, lr, mm, mode)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
            
                    n_train_sample = math.ceil(0.8 * n_sample)
                    n_valid_sample = n_sample - n_train_sample

                    train_data = new_x[:n_train_sample]
                    valid_data = new_x[n_train_sample:]

                    print("total sample:", n_sample,"train sample:", n_train_sample, "valid sample:", n_valid_sample)

                    n_train_batch = math.ceil(n_train_sample/batch_size)
                    n_val_batch = math.ceil(n_valid_sample/batch_size)

                    for n_iter in range(num_iterations):
                        print("\nEPOCH:", str(n_iter+1), "bsz:", str(batch_size), "lr:", str(lr), "mm:", str(mm))
                        total_train_loss = 0
                        total_val_loss = 0

                        # TRAIN
                        print("TRAIN")
                        pbar = tqdm(range(n_train_batch), total=n_train_batch)
                        for i in pbar:
                            batch_input = train_data[i * batch_size:(i+1) * batch_size]

                            feed_dict = {placeholder_x: batch_input}
                            _, l = sess.run([train_op, loss], feed_dict=feed_dict)
                            total_train_loss += l
                            input_batch_size = batch_input.shape[0]

                            pbar.set_description('L:{:.5f}'.format(total_train_loss / (i+ 1)))

                        train_loss = total_train_loss / (i+1)

                        # VALID
                        print("VALID")
                        pbar = tqdm(range(n_val_batch), total=n_val_batch)
                        for i in pbar:
                            batch_input = valid_data[i * batch_size:(i+1) * batch_size]

                            feed_dict = {placeholder_x: batch_input}
                            l = sess.run([loss], feed_dict=feed_dict)
                            total_val_loss += l[0]
                            input_batch_size = batch_input.shape[0]

                            pbar.set_description('L:{:.5f}'.format(total_val_loss / (i+ 1)))

                        val_loss = total_val_loss / n_val_batch
                        print("AVG loss", str(total_val_loss / n_val_batch))

                        train_losses.append(train_loss)
                        val_losses.append(val_loss)

                        if best_config_val_loss > val_loss:
                            et = 0
                            best_config_val_loss = val_loss

                            if best_val_loss > best_config_val_loss:
                                best_val_loss = best_config_val_loss
                                best_batch_size = bsz
                                best_lr = lr
                        else:
                            et += 1

                        if et == 2:
                            print("EARLY STOPPING best config: loss=" + str(best_config_val_loss) + " lr=" + str(lr) + " bsz=" + str(batch_size) + " n_iter=" + str(n_iter+1))
                            break
                
                end_time = time.time()
                elapsed_time = start_time - end_time
                elapsed_times[mode] = elapsed_time
                all_train_losses[mode] = train_losses
                all_val_losses[mode] = val_losses
                modes.append(mode)

                plot_graph("ae_cross_val_train_losses_" + mode + ".pdf", train_losses, "iteration", "train loses", "Train losses " + mode)
                plot_graph("ae_cross_val_val_losses_" + mode + ".pdf", val_losses, "iteration", "val losses", "Val losses " + mode)
                print("mode:", mode, "elapsed time:", elapsed_time,"sec")

    for i in range(len(mode)):
        # print(mode, "train loss:", all_train_losses[mode])
        # print(mode, "train accu:", all_train_accus[mode])
        # print(mode, "val loss:", all_val_losses[mode])
        # print(mode, "val accu:", all_val_accus[mode])
        print(mode, "elapsed time:", elapsed_times[mode])

    print("BEST CONFIG FROM GRID SEARCH:")
    print("loss:", best_val_loss, "batch_size:", best_batch_size, "lr:", best_lr, "momentum:", best_momentum)

def evaluate_ae(x, placeholder_x, batch_size, params, train_op, loss, out, enc_conv3 ):
    print("TEST CNN")

    tf.reset_default_graph()
    # params, train_op, loss, out, enc_conv3 = ae_model(placeholder_x, 0.1, 0.1, "eval-ae")
    # saver = tf.train.Saver()

    # with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    # saver.restore(sess, AE_MODEL_PATH)
    # print("Model restored.")

    n_sample = x.shape[0]
    indices = [x for x in range(n_sample)]
    np.random.shuffle(indices)
    new_x = x[indices]

    n_test_sample = n_sample
    test_data = new_x[:n_test_sample]
    print("total sample:", n_sample,"test sample:", n_test_sample)
    n_test_batch = math.ceil(n_test_sample/batch_size)

    total_test_loss = 0
    
    # TEST
    print("TEST")
    pbar = tqdm(range(n_test_batch), total=n_test_batch)
    for i in pbar:
        batch_input = test_data[i * batch_size:(i+1) * batch_size]

        feed_dict = {placeholder_x: batch_input}
        l = sess.run([loss], feed_dict=feed_dict)
        total_test_loss += l[0]

        pbar.set_description('L:{:.5f}'.format(total_test_loss / (i+1)))

    val_loss = total_test_loss / n_test_batch
    return val_loss

def main():
    parser = argparse.ArgumentParser(description='COMP5212 Programming Project 2')
    parser.add_argument('--task', default="train", type=str,
                        help='Select the task, train_cnn, test_cnn, '
                             'train_ae, evaluate_ae, cross_valid_cnn, cross_valid_ae')
    parser.add_argument('--datapath',default="./data",type=str, required=False,
                        help='Select the path to the data directory')
    parser.add_argument('--lr',default="0.1",type=float, required=False,
                        help='learning rate')
    parser.add_argument('--mm',default="0.2",type=float, required=False,
                        help='momentum')
    parser.add_argument('--bsz',default="0.1",type=float, required=False,
                        help='batch_size')

    args = parser.parse_args()
    datapath = args.datapath
    with tf.variable_scope("placeholders"):
        img_var = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="img")
        label_var = tf.placeholder(tf.int32, shape=(None,), name="true_label")

    if args.task == "train_cnn":
        batch_size = int(args.bsz)
        lr = float(args.lr)
        momentum = float(args.mm)

        file_train = np.load(datapath+"/data_classifier_train.npz")
        x_train = file_train["x_train"]
        y_train = file_train["y_train"]

        train_cnn(x_train, y_train, img_var, label_var, batch_size, momentum, lr)
    elif args.task == "cross_valid_cnn":
        file_train = np.load(datapath+"/data_classifier_train.npz")
        x_train = file_train["x_train"]
        y_train = file_train["y_train"]
        cross_valid_cnn(x_train, y_train, img_var, label_var)
    elif args.task == "test_cnn":
        batch_size = int(args.bsz)
        lr = float(args.lr)
        mm = float(args.mm)

        print("RETRAIN BEFORE EVAL")
        file_train = np.load(datapath+"/data_classifier_train.npz")
        x_train = file_train["x_train"]
        y_train = file_train["y_train"]

        params, train_op, loss, accuracy = train_cnn(x_train, y_train, img_var, label_var, batch_size, mm, lr)

        print("EVAL")
        file_test = np.load(datapath+"/data_classifier_test.npz")
        x_test = file_test["x_test"]
        y_test = file_test["y_test"]
        accuracy = test_cnn(x_test, y_test, img_var, label_var, batch_size, mm, lr, params, train_op, loss, accuracy)
        print("accuracy = {}\n".format(accuracy))
    elif args.task == "train_ae":
        batch_size = int(args.bsz)
        lr = float(args.lr)
        mm = float(args.mm)

        file_unsupervised = np.load(datapath + "/data_autoencoder_train.npz")
        x_ae_train = file_unsupervised["x_ae_train"]
        train_ae(x_ae_train, img_var, batch_size, lr, mm)
    elif args.task == "cross_valid_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_train.npz")
        x_ae_train = file_unsupervised["x_ae_train"]

        cross_valid_ae(x_ae_train, img_var)
    elif args.task == "evaluate_ae":
        batch_size = int(args.bsz)
        lr = float(args.lr)
        mm = float(args.mm)

        print("RETRAIN BEFORE EVAL")
        file_unsupervised = np.load(datapath + "/data_autoencoder_train.npz")
        x_ae_train = file_unsupervised["x_ae_train"]
        params, train_op, loss, out, enc_conv3  = train_ae(x_ae_train, img_var, batch_size, lr, mm)

        print("EVAL")
        file_unsupervised = np.load(datapath + "/data_autoencoder_eval.npz")
        x_ae_eval = file_unsupervised["x_ae_eval"]
        evaluate_ae(x_ae_eval, img_var, batch_size, params, train_op, loss, out, enc_conv3)

if __name__ == "__main__":
    main()
