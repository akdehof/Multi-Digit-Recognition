import os
from datetime import datetime
import time
import tensorflow as tf
import os
import numpy as np
import h5py
import random
from PIL import Image
import tensorflow as tf
import json
from model import Model


 
def build_batch(path_to_tfrecords_file, num_examples, batch_size, shuffled):
    assert tf.gfile.Exists(path_to_tfrecords_file), '%s not found' % path_to_tfrecords_file
    
    filename_queue = tf.train.string_input_producer([path_to_tfrecords_file], num_epochs=None)
    #image, length, digits = read_and_decode(filename_queue)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([5], tf.int64)
            })
    
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.reshape(image, [64, 64, 3])
    image = tf.random_crop(image, [54, 54, 3])
    
    length = tf.cast(features['length'], tf.int32)
    digits = tf.cast(features['digits'], tf.int32)
    
    
    
    min_queue_examples = int(0.4 * num_examples)
    
    image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_examples + 3 * batch_size)
    return image_batch, length_batch, digits_batch


  
#Used for test the accuracy for both training and test process
class Evaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)
    
    
 
    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples // batch_size
        needs_include_length = False
    
        with tf.Graph().as_default():
            image_batch, length_batch, digits_batch = build_batch(path_to_tfrecords_file,
                                                                         num_examples=num_examples,
                                                                         batch_size=batch_size,
                                                                         shuffled=False)
            length_logits, digits_logits = Model.inference(image_batch, drop_rate=0.0)
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)
    
            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions
    
            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)
    
            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )
    
            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()
    
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)
    
                for _ in range(num_batches):
                    sess.run(update_accuracy)
    
                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)
    
                coord.request_stop()
                coord.join(threads)
    
        return accuracy_val
    

#build the model with the tensor flow
def multi_digit_model(input_x, y_len, y_digit, drop_rate):
    with tf.variable_scope('hidden1'):
        conv = tf.layers.conv2d(input_x, filters = 48, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 2, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden1 = dropout
    
    with tf.variable_scope('hidden2'):
        conv = tf.layers.conv2d(hidden1, filters = 64, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 1, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden2 = dropout
    
    with tf.variable_scope('hidden3'):
        conv = tf.layers.conv2d(hidden2, filters = 128, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 2, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden3 = dropout
    
    with tf.variable_scope('hidden4'):
        conv = tf.layers.conv2d(hidden3, filters = 160, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 1, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden4 = dropout
    
    with tf.variable_scope('hidden5'):
        conv = tf.layers.conv2d(hidden4, filters = 192, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 2, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden5 = dropout
    
    with tf.variable_scope('hidden6'):
        conv = tf.layers.conv2d(hidden5, filters = 192, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 1, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden6 = dropout
    
    with tf.variable_scope('hidden7'):
        conv = tf.layers.conv2d(hidden6, filters = 192, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 2, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden7 = dropout
    
    with tf.variable_scope('hidden8'):
        conv = tf.layers.conv2d(hidden7, filters = 192, kernel_size = [5, 5], padding = 'same')
        norm = tf.nn.relu(tf.layers.batch_normalization(conv))
        maxpooling = tf.layers.max_pooling2d(norm, pool_size = [2, 2], strides = 1, padding = 'same')
        dropout = tf.layers.dropout(maxpooling, rate = drop_rate)
        hidden8 = dropout
    
    with tf.variable_scope('flatten'):
        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])
    
    
    with tf.variable_scope('hidden9'):
        hidden9 = tf.layers.dense(flatten, units = 3072, activation = tf.nn.relu)
    
    
    with tf.variable_scope('hidden10'):
        hidden10 = tf.layers.dense(hidden9, units = 3072, activation = tf.nn.relu)
    
    
    with tf.variable_scope('digit_length'):
        length = tf.layers.dense(hidden10, units = 7)
    
    
    with tf.variable_scope('digit1'):
        digit1 = tf.layers.dense(hidden10, units = 11)
    
    
    with tf.variable_scope('digit2'):
        digit2 = tf.layers.dense(hidden10, units = 11)
    
    
    with tf.variable_scope('digit3'):
        digit3 = tf.layers.dense(hidden10, units = 11)
    
    
    with tf.variable_scope('digit4'):
        digit4 = tf.layers.dense(hidden10, units = 11)
    
    # TODO ANNE hier muss es laenger werden
    with tf.variable_scope('digit5'):
        digit5 = tf.layers.dense(hidden10, units = 11)
    
    
    digits = tf.stack([digit1, digit2, digit3, digit4, digit5], axis = 1)
    
    length_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y_len, logits = length))
    d1_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y_digit[:, 0], logits = digits[:, 0, :]))
    d2_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y_digit[:, 1], logits = digits[:, 1, :]))
    d3_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y_digit[:, 2], logits = digits[:, 2, :]))
    d4_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y_digit[:, 3], logits = digits[:, 3, :]))
    d5_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = y_digit[:, 4], logits = digits[:, 4, :]))
    
    total_loss = length_ce + d1_ce + d2_ce + d3_ce + d4_ce + d5_ce
    
    
    return length, digits, total_loss


#Build the training process    
def train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 1
    num_steps_to_check = 10
    
    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = build_batch(path_to_train_tfrecords_file,
                                                                     num_examples=num_train_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
    
        _, _, loss = multi_digit_model(image_batch, length_batch, digits_batch, drop_rate=0.2)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
    
        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()
    
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))
    
            #tf.global_variables_initializer()
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print ('Model restored from file: %s' % path_to_restore_checkpoint_file)
    
            print ('Start training')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0
    
            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time
    
                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print ('%s: step %d, loss = %f ' % (
                        datetime.now(), global_step_val, loss_val))
    
                if global_step_val % num_steps_to_check != 0:
                    continue
    
                summary_writer.add_summary(summary_val, global_step=global_step_val)
    
    
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                print ('Validation accuracy is= %f, best accuracy %f' % (accuracy, best_accuracy))
    
                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print ('Save file to: %s' % path_to_checkpoint_file)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1
    
    
                if patience == 0:
                    break
    
            coord.request_stop()
            coord.join(threads)
            print ('Training progess is finished')

#Build the training process
def train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 1
    num_steps_to_check = 10
    
    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = build_batch(path_to_train_tfrecords_file,
                                                                     num_examples=num_train_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
    
        _, _, loss = multi_digit_model(image_batch, length_batch, digits_batch, drop_rate=0.2)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
    
        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()
    
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))
    
            tf.global_variables_initializer()
            sess.run(tf.global_variables_initializer())
            # TODO
            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print ('Model restored from file: %s' % path_to_restore_checkpoint_file)
    
            print ('Start training')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0 # Anne war auskommentiert
    
            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time # Anne war auskommentiert
    
                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print ('%s: step %d, loss = %f ' % (
                        datetime.now(), global_step_val, loss_val))
    
                if global_step_val % num_steps_to_check != 0:
                    continue
    
                summary_writer.add_summary(summary_val, global_step=global_step_val)
    
    
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                print ('Validation accuracy is= %f, best accuracy %f' % (accuracy, best_accuracy))
    
                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print ('Save file to: %s' % path_to_checkpoint_file)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1
    
    
                if patience == 0:
                    break
    
            coord.request_stop()
            coord.join(threads)
            print ('Training progess is finished')


#
#     M A I N 
#

#def main(_):
#Train the model 
#define the folder path

## TODO
log_dir = './logs/train'
# ANNE
# TODO
path_to_data_dir = 'C:/Users/anne/src/Z-AI-ler/Data/SVHN/'

train_tfrecords_file =  os.path.join(path_to_data_dir , '/data/train.tfrecords')
val_tfrecords_file = os.path.join(path_to_data_dir ,    '/data/val.tfrecords')
tfrecords_meta_file = os.path.join(path_to_data_dir ,   '/data/meta.json')



restore_checkpoint_file = None
opt = {
    'batch_size': 32,
    'learning_rate': 1e-2,
    'patience': 100,
    'decay_steps': 10000,
    'decay_rate': 0.9
    }
with open(tfrecords_meta_file, 'r') as f:
    content = json.load(f)
    num_train_examples = content['num_examples']['train']
    num_val_examples = content['num_examples']['val']
    num_test_examples = content['num_examples']['test']
    
    
#train the model     
train(train_tfrecords_file, num_train_examples,
        val_tfrecords_file, num_val_examples,
        log_dir, restore_checkpoint_file,
        opt)

