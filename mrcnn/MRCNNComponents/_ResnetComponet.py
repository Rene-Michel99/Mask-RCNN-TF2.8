import tensorflow as tf
import tensorflow.keras.layers as KL

from mrcnn.Configs import Config
from mrcnn.CustomLayers import BatchNorm


class ResnetComponent:
    def __init__(self, input_image: KL.Input, config: Config):
        self.rpn_feature_maps = None
        self.mrcnn_feature_maps = None

        self.build(input_image, config)

    def build(self, input_image: KL.Input, config: Config):
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
            _, c2, c3, c4, c5 = config.BACKBONE(
                input_image, stage5=True,
                train_bn=config.TRAIN_BN
            )
        else:
            _, c2, c3, c4, c5 = self._resnet_graph(
                input_image, config.BACKBONE,
                stage5=True, train_bn=config.TRAIN_BN
            )

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
        self.rpn_feature_maps = [p2, p3, p4, p5, p6]
        self.mrcnn_feature_maps = [p2, p3, p4, p5]

    def _resnet_graph(self, input_image: KL.Input, architecture: str, stage5=False, train_bn=True):
        """Build a ResNet graph.

        Params:
            - architecture: Can be resnet50 or resnet101
            - stage5: Boolean. If False, stage5 of the network is not created
            - train_bn: Boolean. Train or freeze Batch Norm layers
        """
        assert architecture in ["resnet50", "resnet101"]

        # Stage 1
        x = KL.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(input_image)
        x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = BatchNorm(name='bn_conv1', epsilon=1.001e-5)(x, training=train_bn)  # noqa
        x = KL.Activation('relu')(x)
        C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = self._conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
        # Stage 3
        x = self._conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
        # Stage 4
        x = self._conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x
        # Stage 5
        if stage5:
            x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]

    @staticmethod
    def _conv_block(input_tensor, kernel_size, filters, stage, block,
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
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)  # noqa
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)  # noqa
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)  # noqa

        shortcut = KL.Conv2D(
            nb_filter3, (1, 1), strides=strides,
            name=conv_name_base + '1', use_bias=use_bias
        )(input_tensor)
        shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)  # noqa

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    @staticmethod
    def _identity_block(input_tensor, kernel_size, filters, stage, block,
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
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)  # noqa
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)  # noqa
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)  # noqa

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x
