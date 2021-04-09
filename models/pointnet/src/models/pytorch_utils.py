import numpy as np
import torch
import torch.nn as nn

# def gather_nd(params, indices):
#
#     # print("Inside gather_nd pytorch")
#     # print("params shape")
#     # print(params.shape)
#     # print("indices shape")
#     # print(indices.shape)
#     out_shape = indices.shape
#     # print("outshape")
#     # print(out_shape)
#     indices = indices.unsqueeze(-1).transpose(0, -1)  # roll last axis to fring
#     ndim = indices.shape[0]
#     # print("indices shape")
#     # print(indices.shape)
#     # print("ndim")
#     # print(ndim)
#     indices = indices.long()
#     # print("indices shape")
#     # print(indices.shape)
#     # print("indices[0] shape")
#     # print(indices[0].shape)
#     idx = torch.zeros_like(indices[0], device=indices.device).long()
#     # print("idx shape")
#     # print(idx.shape)
#     m = 1
#
#     for i in range(ndim)[::-1]:
#         idx += indices[i] * m
#         m *= params.size(i)
#     # print("idx shape just before torch.take call")
#     # print(idx.shape)
#     out = torch.take(params, idx)
#     return out.view(out_shape)

def gather_nd(params, indices):
    out = params[list(indices.T)]
    return out


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
           stride=1,
           padding=0,
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=nn.ReLU(),
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
    data_format: 'NHWC' or 'NCHW'
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
      num_in_channels = inputs.size(-1)
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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  conv = nn.Conv1d(num_in_channels, num_output_channels, kernel_size, stride, padding).to(device)

  if data_format == 'NHWC':
    #inputs = inputs.transpose(2, -1).transpose(1, 2)
    inputs = inputs.transpose(1, -1).transpose(2, -1)

  # biases = _variable_on_cpu('biases', [num_output_channels],
  #                         tf.constant_initializer(0.0))

  # biases = torch.Tensor([0.0] * num_output_channels, dtype=torch.float32)
  # outputs = outputs + biases

  # outputs = tf.nn.bias_add(outputs, biases)

  outputs = conv(inputs.to(device))

  outputs = outputs.to(device)
  if bn:
      # outputs = batch_norm_for_conv2d(outputs, is_training,
      #                                 bn_decay=bn_decay, scope='bn')
      num_features = outputs.size(1)

      if is_training:
          if bn_decay is None:
              batch_norm = nn.BatchNorm1d(num_features).to(device).train()
          else:
              batch_norm = nn.BatchNorm1d(num_features, momentum=1 - bn_decay).to(device).train()

      else:
          if bn_decay is None:
              batch_norm = nn.BatchNorm1d(num_features).to(device).eval()
          else:
              batch_norm = nn.BatchNorm1d(num_features, momentum=1 - bn_decay).to(device).eval()

      outputs = batch_norm(outputs)

  if activation_fn is not None:
    # relu = nn.ReLU()
    # outputs = relu(outputs)
    outputs = activation_fn(outputs)
  if data_format == 'NHWC':
    outputs = outputs.transpose(1, 2).transpose(2, -1)
  return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           stride=[1, 1],
           padding=0,
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=nn.ReLU(),
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
    data_format: 'NHWC' or 'NCHW'
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
      num_in_channels = inputs.size(-1)
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
  # print("num_in_channels")
  # print(num_in_channels)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # if torch.cuda.is_available():
  #   conv = nn.Conv2d(num_in_channels, num_output_channels, kernel_size, stride, padding).cuda()
  # else:
  #   conv = nn.Conv2d(num_in_channels, num_output_channels, kernel_size, stride, padding)

  conv = nn.Conv2d(num_in_channels, num_output_channels, kernel_size, stride, padding).to(device)

  # biases = _variable_on_cpu('biases', [num_output_channels],
  #                           tf.constant_initializer(0.0))
  #
  #
  # outputs = tf.nn.bias_add(outputs, biases)

  if data_format == 'NHWC':
    #inputs = inputs.transpose(2, -1).transpose(1, 2)
    #2 512 12 9
    inputs = inputs.transpose(1, -1).transpose(2, -1)

  outputs = conv(inputs.to(device))
  # print("outputs shape")
  # print(outputs.shape)

  outputs = outputs.to(device)
  if bn:
    # outputs = batch_norm_for_conv2d(outputs, is_training,
    #                                 bn_decay=bn_decay, scope='bn')
    num_features = outputs.size(1)
    # print("num_features")
    # print(num_features)
    if is_training:
        if bn_decay is None:
            batch_norm = nn.BatchNorm2d(num_features).to(device).train()
        else:
            batch_norm = nn.BatchNorm2d(num_features, momentum=1 - bn_decay).to(device).train()

    else:
        if bn_decay is None:
            batch_norm = nn.BatchNorm2d(num_features).to(device).eval()
        else:
            batch_norm = nn.BatchNorm2d(num_features, momentum=1 - bn_decay).to(device).eval()
    outputs = batch_norm(outputs)

  if activation_fn is not None:
    # relu = nn.ReLU()
    # outputs = relu(outputs)
    outputs = activation_fn(outputs)
  if data_format == 'NHWC':
    outputs = outputs.transpose(1, 2).transpose(2, -1)
  return outputs