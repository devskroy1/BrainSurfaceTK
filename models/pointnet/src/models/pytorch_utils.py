import numpy as np
import torch
import torch.nn as nn

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with torch.device('cpu:0'):
    dtype = torch.float16 if use_fp16 else torch.float32
    zeros = torch.zeros(shape, dtype=dtype)
    var = initializer(zeros)
    #var = torch.Tensor(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer
  Returns:
    Variable Tensor
  """
  if use_xavier:
    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = nn.init.xavier_uniform_()
  else:
    #initializer = tf.truncated_normal_initializer(stddev=stddev)
    initializer = nn.init.trunc_normal_(std=stddev)

  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = torch.multiply(0.5 * torch.sum(torch.square(var)), wd)
    #tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding=0,
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=nn.ReLU,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.
  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: int
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  #with tf.variable_scope(scope) as sc:
  assert (data_format == 'NHWC' or data_format == 'NCHW')
  if data_format == 'NHWC':
      #num_in_channels = inputs.get_shape()[-1].value
      num_in_channels = inputs.size(3)
  elif data_format == 'NCHW':
      #num_in_channels = inputs.get_shape()[1].value
      num_in_channels = inputs.size(1)
  # kernel_shape = [kernel_size,
  #                   num_in_channels, num_output_channels]
  # kernel = _variable_with_weight_decay('weights',
  #                                        shape=kernel_shape,
  #                                        use_xavier=use_xavier,
  #                                        stddev=stddev,
  #                                        wd=weight_decay)

  conv = nn.Conv1d(num_in_channels, num_output_channels, kernel_size, stride, padding)

  # biases = _variable_on_cpu('biases', [num_output_channels],
  #                         tf.constant_initializer(0.0))

  # biases = torch.Tensor([0.0] * num_output_channels, dtype=torch.float32)
  # outputs = outputs + biases

  # outputs = tf.nn.bias_add(outputs, biases)

  outputs = conv(inputs)
  if bn:
      # outputs = batch_norm_for_conv2d(outputs, is_training,
      #                                 bn_decay=bn_decay, scope='bn')
      num_features = outputs.size(1)
      if is_training:
          batch_norm = nn.BatchNorm1d(num_features, momentum=1 - bn_decay).train()
      else:
          batch_norm = nn.BatchNorm1d(num_features, momentum=1 - bn_decay).eval()
      outputs = batch_norm(outputs)

  if activation_fn is not None:
    outputs = activation_fn(outputs)
  return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding=0,
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=nn.ReLU,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.
  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: int
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  #with tf.variable_scope(scope) as sc:
  # kernel_h, kernel_w = kernel_size
  assert (data_format == 'NHWC' or data_format == 'NCHW')
  if data_format == 'NHWC':
      #num_in_channels = inputs.get_shape()[-1].value
      num_in_channels = inputs.size(3)
  elif data_format == 'NCHW':
      #num_in_channels = inputs.get_shape()[1].value
      num_in_channels = inputs.size(1)

  # kernel_shape = [kernel_h, kernel_w,
  #                 num_in_channels, num_output_channels]

  # kernel = _variable_with_weight_decay('weights',
  #                                      shape=kernel_shape,
  #                                      use_xavier=use_xavier,
  #                                      stddev=stddev,
  #                                      wd=weight_decay)
  # stride_h, stride_w = stride
  conv = nn.Conv2d(num_in_channels, num_output_channels, kernel_size, stride, padding)
  # biases = _variable_on_cpu('biases', [num_output_channels],
  #                           tf.constant_initializer(0.0))
  #
  #
  # outputs = tf.nn.bias_add(outputs, biases)
  outputs = conv(inputs)
  if bn:
    # outputs = batch_norm_for_conv2d(outputs, is_training,
    #                                 bn_decay=bn_decay, scope='bn')
    num_features = outputs.size(1)
    if is_training:
        batch_norm = nn.BatchNorm2d(num_features, momentum=1 - bn_decay).train()
    else:
        batch_norm = nn.BatchNorm2d(num_features, momentum=1 - bn_decay).eval()
    outputs = batch_norm(outputs)

  if activation_fn is not None:
    outputs = activation_fn(outputs)
  return outputs