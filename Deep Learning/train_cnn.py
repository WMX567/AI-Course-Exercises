import tensorflow as tf
import utils
from data.food import FoodDataset
from model.build_cnn import cnn
from train_common import *
from utils import get

def report_training_progress(
	sess, batch_index, images, labels, loss, acc, food):
	 if batch_index % 50 == 0:
	 	batch_images, batch_labels = food.get_batch(
	 		partition='valid', batch_size=512)
	 	valid_acc, valid_loss = sess.run([acc, loss],
	 	feed_dict={images : batch_images, labels : batch_labels})
	 	utils.log_training(batch_index, valid_loss, valid_acc)
	 	utils.update_training_plot(batch_index, valid_acc, valid_loss)


def train_cnn(
 sess, saver, save_path, images, labels, loss, train_op, acc, food):
 """
 Trains a tensorflow model of a cnn to classify a labeled image dataset.
 Periodically saves model checkpoints and reports the network
 performance on a validation set. """
 utils.make_training_plot()
 for batch_index in range(get('cnn.num_steps')):
 	report_training_progress(sess, batch_index, images, labels, loss, acc, food)
 	# Run one step of training
 	batch_images, batch_labels = food.get_batch(partition='train', batch_size=get('cnn.batch_size'))
 	sess.run(train_op, feed_dict={images: batch_images, labels: batch_labels})
 	# Save model parameters periodically
 	if batch_index % 50 == 0:
 		saver.save(sess, save_path)

 	print('CNN Accuracy')
 	for label in range(0, 4):
 		class_images, class_labels = get_class_examples(food, label)
 		accuracy = sess.run(acc, feed_dict={images: class_images, labels: class_labels})
 		print('Class {}: {}'.format(food.get_semantic_label(label), accuracy))
 		saver.save(sess, save_path)

def main():
	print('building model...')
	images, labels = supervised_placeholders()
	logits = cnn(images)
	acc = accuracy(labels, logits)
	loss = cross_entropy_loss(labels, logits)
	train_op = supervised_optimizer(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver, save_path = utils.restore(sess, get('cnn.checkpoint'))
		food = FoodDataset(get('cnn.num_classes'))
		train_cnn(sess, saver, save_path, images, labels, loss, train_op, acc, food)
		print('saving trained model...\n')
		saver.save(sess, save_path)
		utils.hold_training_plot()

if __name__ == '__main__':
	main()