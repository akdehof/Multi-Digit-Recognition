import os
import tensorflow as tf
from model import Model
import json

 
#Data cleaning and preprocessing, read image and convert to training batch
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


#Used for evaluate the accuracy of the model
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



#
#     M A I N 
#


#def main(_):
## TODO
log_dir = './logs/ '
# ANNE
# TODO
path_to_data_dir = 'C:/Users/anne/src/Z-AI-ler/Data/SVHN/'

# start the testing progress
#path_to_train_tfrecords_file =  os.path.join(path_to_data_dir , '/data/train.tfrecords')
#path_to_val_tfrecords_file =    os.path.join(path_to_data_dir , '/data/val.tfrecords')
path_to_test_tfrecords_file =   os.path.join(path_to_data_dir , '/data/test.tfrecords')
path_to_tfrecords_meta_file =   os.path.join(path_to_data_dir , '/data/meta.json')
path_to_checkpoint_dir =        os.path.join(log_dir , '/train')
#path_to_train_eval_log_dir =    os.path.join(log_dir , '/eval/train')
#path_to_val_eval_log_dir =      os.path.join(log_dir , '/eval/val')
path_to_test_eval_log_dir =     os.path.join(log_dir , '/eval/test')


with open(path_to_tfrecords_meta_file, 'r') as f:
        content = json.load(f)
        num_train_examples = content['num_examples']['train']
        num_val_examples = content['num_examples']['val']
        num_test_examples = content['num_examples']['test']
    

evaluator = Evaluator(path_to_test_eval_log_dir)

checkpoint_paths = tf.train.get_checkpoint_state(path_to_checkpoint_dir).all_model_checkpoint_paths

for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
    
    try:
        global_step_val = int(global_step)   
        
    except ValueError:
        continue

        
    accuracy = evaluator.evaluate(path_to_checkpoint, path_to_test_tfrecords_file, num_test_examples,
                                    global_step_val)
    print ('Evaluate the model %s on %s, test accuracy is = %f' \
    % (path_to_checkpoint, path_to_test_tfrecords_file, accuracy))


 