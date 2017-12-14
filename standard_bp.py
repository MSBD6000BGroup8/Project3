import numpy as np
import tensorflow as tf
import os
import tables
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=192, kernel_size=[5, 5], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)

    cccp1 = tf.layers.conv2d(inputs=conv1, filters=160, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)

    cccp2 = tf.layers.conv2d(inputs=cccp1, filters=96, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=cccp2, pool_size=[3, 3], strides=2, padding='same')
    pool1 = tf.layers.dropout(inputs=pool1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=192, kernel_size=[5, 5], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)

    cccp3 = tf.layers.conv2d(inputs=conv2, filters=192, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)

    cccp4 = tf.layers.conv2d(inputs=cccp3, filters=192, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)
    # Pooling Layer #2
    pool2 = tf.layers.average_pooling2d(inputs=cccp4, pool_size=[3, 3], strides=2, padding='same')
    pool2 = tf.layers.dropout(inputs=pool2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(inputs=pool2, filters=192, kernel_size=[3, 3], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)

    cccp5 = tf.layers.conv2d(inputs=conv3, filters=192, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)

    cccp6 = tf.layers.conv2d(inputs=cccp5, filters=10, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu, kernel_initializer=None)
    # Pooling Layer #3
    pool3 = tf.layers.average_pooling2d(inputs=cccp6, pool_size=[8, 8], strides=1, padding="valid")

    # Flatten tensor into a batch of vectors
    pool3_flat = tf.reshape(pool3, [-1, 10])

    # Logits layer
    pool3_flat = tf.squeeze(pool3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training, eval and test data
    hdf5_path = 'cifar10.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='r')

    train_data = np.asarray(hdf5_file.root.train_img)
    train_labels = np.asarray(hdf5_file.root.train_labels)

    eval_data = np.asarray(hdf5_file.root.val_img)
    eval_labels = np.asarray(hdf5_file.root.val_labels)

    test_data = np.asarray(hdf5_file.root.test_img)
    test_labels = np.asarray(hdf5_file.root.test_labels)

    # Create the Estimator
    image_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./model")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    Results = pd.DataFrame(columns=['Epoch','Test_Accuracy'])
    for i in range(251):
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=128,
            num_epochs=None,
            shuffle=True)
        image_classifier.train(
            input_fn=train_input_fn,
            steps=1)

        # Monitor the validation accuarcy every iteration
        if i % 1 == 0:
            # eval_results = image_classifier.evaluate(input_fn=eval_input_fn)
            # print("Validation result: \n",eval_results)
            test_results = image_classifier.evaluate(input_fn=test_input_fn)
            print("Test result: \n",test_results)
            Results.loc[len(Results),'Epoch'] = i
            Results.loc[len(Results)-1, 'Test_Accuracy'] = test_results['accuracy']

    Results.to_csv('Results.csv',index=False)



if __name__ == "__main__":
    filelist = [f for f in os.listdir('./model') if f != 'eval']
    for f in filelist:
        os.remove(os.path.join('./model/', f))
    tf.app.run()

