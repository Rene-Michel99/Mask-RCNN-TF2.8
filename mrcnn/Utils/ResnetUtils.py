import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from mrcnn.CustomLayers import BatchNorm
from .RpnUtils import rpn_graph


# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    Params:
    - anchors_per_location: number of anchors per pixel in the feature map
    - anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    - depth: Depth of the backbone feature map.

    Returns: a Keras Model object. The model outputs, when called, are:
    - rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    - rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut

    Params:
        - input_tensor: input tensor
        - kernel_size: default 3, the kernel size of middle conv layer at main path
        - filters: list of integers, the nb_filters of 3 conv layer at main path
        - stage: integer, current stage label, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names
        - use_bias: Boolean. To use or not use a bias in conv layers.
        - train_bn: Boolean. Train or freeze Batch Norm layers

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn) # noqa
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn) # noqa
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn) # noqa

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn) # noqa

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut

    Params:
        - input_tensor: input tensor
        - kernel_size: default 3, the kernel size of middle conv layer at main path
        - filters: list of integers, the nb_filters of 3 conv layer at main path
        - stage: integer, current stage label, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names
        - use_bias: Boolean. To use or not use a bias in conv layers.
        - train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn) # noqa
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn) # noqa
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn) # noqa

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


############################################################
#  Resnet Graph
############################################################
def block2(x, filters, kernel_size=3, first_layers_stride=1, conv_shortcut=False, name=None, train_bn=False):
    """A residual block.

    Parms:
        - x: input tensor.
        - filters: integer, filters of the bottleneck layer.
        - kernel_size: default 3, kernel size of the bottleneck layer.
        - first_layers_stride: default 1, stride of the first layer.
        - conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        - name: string, block label.
        - train_bn: bool

    Returns: Output tensor for the residual block.
    """

    preact = BatchNorm(name=name + '_bn2', epsilon=1.001e-5)(x, training=train_bn) # noqa
    preact = KL.Activation("relu", name=name + "_preact_relu")(preact)

    if conv_shortcut:
        shortcut = KL.Conv2D(
            4 * filters, 1, strides=first_layers_stride, name=name + "_0_conv"
        )(preact)
    else:
        shortcut = (
            KL.MaxPooling2D(1, strides=first_layers_stride)(x) if first_layers_stride > 1 else x
        )

    x = KL.Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + "_1_conv"
    )(preact)
    x = BatchNorm(name=name + '_bn3', epsilon=1.001e-5)(x, training=train_bn) # noqa
    x = KL.Activation("relu", name=name + "_1_relu")(x)

    x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = KL.Conv2D(
        filters,
        kernel_size,
        strides=first_layers_stride,
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = BatchNorm(name=name + '_bn4', epsilon=1.001e-5)(x, training=train_bn) # noqa
    x = KL.Activation("relu", name=name + "_2_relu")(x)

    x = KL.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = KL.Add(name=name + "_out")([shortcut, x])
    return x


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.

    Params:
      - x: input tensor.
      - filters: integer, filters of the bottleneck layer.
      - kernel_size: default 3, kernel size of the bottleneck layer.
      - stride: default 1, stride of the first layer.
      - conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      - name: string, block label.

    Returns: Output tensor for the residual block.
    """
    bn_axis = 3

    if conv_shortcut:
        shortcut = KL.Conv2D(
            4 * filters, 1, strides=stride, name=name + "_0_conv"
        )(x)
        shortcut = KL.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = KL.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = KL.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn"
    )(x)
    x = KL.Activation("relu", name=name + "_1_relu")(x)

    x = KL.Conv2D(
        filters, kernel_size, padding="SAME", name=name + "_2_conv"
    )(x)
    x = KL.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn"
    )(x)
    x = KL.Activation("relu", name=name + "_2_relu")(x)

    x = KL.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = KL.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn"
    )(x)

    x = KL.Add(name=name + "_add")([shortcut, x])
    x = KL.Activation("relu", name=name + "_out")(x)
    return x


def stack2(x, filters, blocks, first_layers_stride=1, stride=2, name=None, train_bn=False):
    """A set of stacked residual blocks.

    Params:
      - x: input tensor.
      - filters: integer, filters of the bottleneck layer in a block.
      - blocks: integer, blocks in the stacked blocks.
      - stride1: default 2, stride of the first layer in the first block.
      - name: string, stack label.
      - train_bn: bool, train or not BatchNormalization

    Returns: Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + "_block1", train_bn=train_bn)
    for i in range(2, blocks):
        x = block2(x, filters, name=name + "_block" + str(i), train_bn=train_bn)
    x = block2(x, filters, first_layers_stride=first_layers_stride, name=name + "_block" + str(blocks), train_bn=train_bn)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Params:
      - x: input tensor.
      - filters: integer, filters of the bottleneck layer in a block.
      - blocks: integer, blocks in the stacked blocks.
      - stride1: default 2, stride of the first layer in the first block.
      - name: string, stack label.

    Returns: Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, conv_shortcut=True, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(
            x, filters, name=name + "_block" + str(i)
        )
    return x


def resnet101v2_graph(input_image, train_bn, use_bias=True, stage5=True):
    # Stage 1
    x = KL.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(input_image)
    x = KL.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1")(x)
    x = BatchNorm(name='bn_conv1', epsilon=1.001e-5)(x, training=train_bn)  # noqa
    x = KL.Activation('relu', name="conv1_relu")(x)
    x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    c1 = KL.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    '''C2 = stack2(C1, 64, 3, name="conv2", train_bn=train_bn)
    C3 = stack2(C2, 128, 4, name="conv3", train_bn=train_bn)
    C4 = stack2(C3, 256, 23, name="conv4", train_bn=train_bn)
    if stage5:
        C5 = stack2(C4, 512, 3, first_layers_stride=1, name="conv5", train_bn=train_bn)
    else:
        C5 = None'''
    c2 = stack1(c1, 64, 3, stride1=1, name="conv2")
    c3 = stack1(c2, 128, 4, name="conv3")
    c4 = stack1(c3, 256, 23, name="conv4")
    if stage5:
        c5 = stack1(c4, 512, 3, name="conv5")
    else:
        c5 = None

    return [c1, c2, c3, c4, c5]


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.

    Params:
        - architecture: Can be resnet50 or resnet101
        - stage5: Boolean. If False, stage5 of the network is not created
        - train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101", "resnet101V2"]
    if architecture == "resnet101V2":
        return resnet101v2_graph(input_image, train_bn, stage5=stage5)

    # Stage 1
    x = KL.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1', epsilon=1.001e-5)(x, training=train_bn) # noqa
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def build_resnet_backbone(input_image, config):
    """Similar to the ConvNet that we use in Faster R-CNN to extract feature maps from the image,
    we use the ResNet101 architecture to extract features from the images in Mask R-CNN. So,
    the first step is to take an image and extract features using the ResNet 101 architecture.
    These features act as an input for the next layer.

        Params:
        - input_image: Input layer of tensorflow for image

        Returns: Tuple of [rpn_feature_maps, mrcnn_feature_maps]
            - rpn_feature_maps: List of convolutional layers
            - mrcnn_feature_maps: List of convolutional layers
    """

    # Bottom-up Layers
    # Returns a list of the last layers of each stage, 5 in total.
    # Don't create the thead (stage 5), so we pick the 4th item in the list.
    if callable(config.BACKBONE):
        _, c2, c3, c4, c5 = config.BACKBONE(input_image, stage5=True,
                                            train_bn=config.TRAIN_BN)
    else:
        _, c2, c3, c4, c5 = resnet_graph(input_image, config.BACKBONE,
                                         stage5=True, train_bn=config.TRAIN_BN)

    # Top-down Layers
    p5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(c5)
    assert config.TOP_DOWN_PYRAMID_SIZE == p5.shape[-1]
    size = (2, 2) if config.BACKBONE == "resnet101V2" else (2, 2)
    p4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=size, name="fpn_p5upsampled", interpolation=config.INTERPOLATION_METHOD)(p5),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(c4)
    ])
    p3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled", interpolation=config.INTERPOLATION_METHOD)(p4),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(c3)
    ])
    p2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled", interpolation=config.INTERPOLATION_METHOD)(p3),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(c2)
    ])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    p2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(p2)
    p3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(p3)
    p4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(p4)
    p5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(p5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    p6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(p5)

    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [p2, p3, p4, p5, p6]
    mrcnn_feature_maps = [p2, p3, p4, p5]
    return rpn_feature_maps, mrcnn_feature_maps
