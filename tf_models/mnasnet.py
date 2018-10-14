# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Mobilenet V2.

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

import tensorflow as tf

from tf_models import conv_blocks as ops
from tf_models import mobilenet as lib

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


# pyformat: enable


@slim.add_arg_scope
def mnasnet(input_tensor, num_classes=1001, depth_multiplier=1.0, scope='MobilenetV2', conv_defs=None,
            finegrain_classification_mode=False, min_depth=None, divisible_by=None, **kwargs):

	if conv_defs is None:
		print("Using MNAS default definitions")
		conv_defs = __MNAS_DEF
	if 'multiplier' in kwargs:
		raise ValueError('mobilenetv2 doesn\'t support generic '
		                 'multiplier parameter use "depth_multiplier" instead.')
	if finegrain_classification_mode:
		conv_defs = copy.deepcopy(conv_defs)
		if depth_multiplier < 1:
			conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier

	depth_args = {}
	# NB: do not set depth_args unless they are provided to avoid overriding
	# whatever default depth_multiplier might have thanks to arg_scope.
	if min_depth is not None:
		depth_args['min_depth'] = min_depth
	if divisible_by is not None:
		depth_args['divisible_by'] = divisible_by

	with slim.arg_scope((lib.depth_multiplier,), **depth_args):
		print(conv_defs["spec"][0])
		return lib.mobilenet(input_tensor, num_classes=num_classes, conv_defs=conv_defs, scope=scope,
		                     multiplier=depth_multiplier, **kwargs)


def wrapped_partial(func, *args, **kwargs):
	partial_func = functools.partial(func, *args, **kwargs)
	functools.update_wrapper(partial_func, func)
	return partial_func


# Wrappers for mobilenet v2 with depth-multipliers. Be noticed that
# 'finegrain_classification_mode' is set to True, which means the embedding
# layer will not be shrinked when given a depth-multiplier < 1.0.
mobilenet_v2_140 = wrapped_partial(mnasnet, depth_multiplier=1.4)
mobilenet_v2_050 = wrapped_partial(mnasnet, depth_multiplier=0.50, finegrain_classification_mode=True)
mobilenet_v2_035 = wrapped_partial(mnasnet, depth_multiplier=0.35, finegrain_classification_mode=True)


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
	"""Creates base of the mobilenet (no pooling and no logits) ."""
	return mnasnet(input_tensor, depth_multiplier=depth_multiplier, base_only=True, **kwargs)


def training_scope(**kwargs):
	"""Defines MobilenetV2 training scope.

	Usage:
	   with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
		 logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

	with slim.

	Args:
	  **kwargs: Passed to mobilenet.training_scope. The following parameters
	  are supported:
		weight_decay- The weight decay to use for regularizing the model.
		stddev-  Standard deviation for initialization, if negative uses xavier.
		dropout_keep_prob- dropout keep probability
		bn_decay- decay for the batch norm moving averages.

	Returns:
	  An `arg_scope` to use for the mobilenet v2 model.
	"""
	return lib.training_scope(**kwargs)


__all__ = ['training_scope', 'mobilenet_base', 'mnasnet', '__MNAS_DEF']


if __name__ == '__main__':
	resolution = 224
	num_filters = 3
	dsize = (1, resolution, resolution, num_filters)

	import os, os.path as osp

	with tf.Graph().as_default():
		data = tf.placeholder(tf.float32, shape=dsize, name="input")
		net, end_points = mnasnet(data, num_classes=1000, conv_defs=__MNAS_DEF, scope=None, depth_multiplier=1.0)

		target= "temp"
		with tf.Session() as sess:
			# if osp.exists(target):
			# 	shutil.rmtree(target)
			os.makedirs(target, exist_ok=True)

			writer = tf.summary.FileWriter(target, sess.graph)
			sess.run(tf.global_variables_initializer())
			# fake_data = np.zeros(dsize)
			# sess.run(net, feed_dict={data: fake_data})
			writer.close()

			# net = tf.graph_util.remove_training_nodes(net)
			converter = tf.contrib.lite.TocoConverter.from_session(sess, [data], [net])
			tflite_model = converter.convert()
			model_file = osp.join(target, "converted_model.tflite")
			open(model_file, "wb").write(tflite_model)

