import tensorflow as tf


from tf_models import conv_blocks as ops
from tf_models import mobilenet as lib
from tf_models.mnasnet import mnasnet

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor

# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
__MNAS_DEF = dict(
	defaults={
		# Note: these parameters of batch norm affect the architecture
		# that's why they are here and not in training_scope.
		(slim.batch_norm,): {'center': True, 'scale': True},
		(slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
			'normalizer_fn': slim.batch_norm,
			'activation_fn': tf.nn.relu6
		},
		(ops.expanded_conv,): {
			'expansion_size': expand_input(6),
			'split_expansion': 1,
			'normalizer_fn': slim.batch_norm,
			'residual': True
		},
		(slim.conv2d, slim.separable_conv2d): {
			'padding': 'SAME'
		}
	},
	spec=[
		# 224x224x3 -> 112x112x32
		op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),

		# 112x112x32 -> 112x112x16
		op(ops.expanded_conv, expansion_size=expand_input(1, divisible_by=1), num_outputs=16, residual=False),

		# 112x112x16 -> 56x56x24
		op(ops.expanded_conv, expansion_size=expand_input(3, divisible_by=1), kernel_size=(3, 3), stride=2, num_outputs=24, residual=False),
		op(ops.expanded_conv, expansion_size=expand_input(3, divisible_by=1), kernel_size=(3, 3), stride=1, num_outputs=24, residual=True),
		op(ops.expanded_conv, expansion_size=expand_input(3, divisible_by=1), kernel_size=(3, 3), stride=1, num_outputs=24, residual=True),

		# 56x56x24 -> 28x28x40
		op(ops.expanded_conv, expansion_size=expand_input(3, divisible_by=1), kernel_size=(5, 5), stride=2, num_outputs=40, residual=False),
		op(ops.expanded_conv, expansion_size=expand_input(3, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=40, residual=True),
		op(ops.expanded_conv, expansion_size=expand_input(3, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=40, residual=True),

		# 28x28x40 -> 14x14x80
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=2, num_outputs=80, residual=False),
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=80, residual=True),
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=80, residual=True),

		# 14x14x80 -> 14x14x96
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(3, 3), stride=1, num_outputs=96, residual=False),
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(3, 3), stride=1, num_outputs=96, residual=True),

		# 14x14x96 -> 7x7x192
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=2, num_outputs=192, residual=False),
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=192, residual=True),
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=192, residual=True),
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(5, 5), stride=1, num_outputs=192, residual=True),

		# 7x7x192 -> 7x7x320
		op(ops.expanded_conv, expansion_size=expand_input(6, divisible_by=1), kernel_size=(3, 3), stride=1, num_outputs=320, residual=False),

		op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
	],
)

import shutil
if __name__ == '__main__':
	resolution = 224
	num_filters = 3
	dsize = (1, resolution, resolution, num_filters)

	import os, os.path as osp

	from tf_models.mobilenet_v2 import mobilenetv2, mobilenet_v2_140
	with tf.Graph().as_default():
		data = tf.placeholder(tf.float32, shape=dsize, name="input")
		net, end_points = mnasnet(data, num_classes=1000, scope=None, depth_multiplier=1.0)
		# net = mobilenet_v2_140(data, num_classes=1000, scope=None)

		target = "tmp"
		os.makedirs(target, exist_ok=True)

		with tf.Session() as sess:
			writer = tf.summary.FileWriter(target, sess.graph)
			sess.run(tf.global_variables_initializer())
			writer.close()
			print("Computation graph dumped.")
			# net = tf.graph_util.remove_training_nodes(net)
			converter = tf.contrib.lite.TocoConverter.from_session(sess, [data], [net])
			tflite_model = converter.convert()
			model_file = osp.join(target, "model.tflite")
			if osp.exists(model_file):
				os.remove(model_file)
			open(model_file, "wb").write(tflite_model)
			print("TF-Lite generated.")