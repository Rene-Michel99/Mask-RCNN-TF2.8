import tensorflow as tf
from typing import List

from ._Common import resize_and_crop
from ._Interface import Interface
from mrcnn.Utils.utilfunctions import parse_image_meta_graph


############################################################
#  ROIAlign Layer
############################################################
class PyramidInterface(Interface):
    def __init__(self, pool_shape, interpolation_method):
        self.POOL_SHAPE = pool_shape
        self.INTERPOLATION_METHOD = interpolation_method


class PyramidROIAlign(tf.keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape: List[int], interpolation_method: str, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.interface = PyramidInterface(tuple(pool_shape), interpolation_method)

    @tf.function
    def call(self, inputs):
        """ The input is a list of params

        Params:
        - boxes: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - feature_maps: List of feature maps from different layers of the pyramid,
          [P2, P3, P4, P5]. Each has a different resolution.


        Returns: Pooled regions in the shape [batch, num_boxes, pool_height, pool_width, channels],
            The width and height are those specific in the pool_shape in the layer constructor.

        """
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]
        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]
        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(
            5, tf.maximum(
                2, 4 + tf.cast(tf.round(roi_level), tf.int32)
            )
        )
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            if self.interface.INTERPOLATION_METHOD == "bicubic":
                resized_cropped = resize_and_crop(
                    box_indices,
                    level_boxes,
                    feature_maps[i],
                    self.interface.POOL_SHAPE,
                    method=self.interface.INTERPOLATION_METHOD
                )
            else:
                resized_cropped = tf.image.crop_and_resize(
                    image=feature_maps[i],
                    boxes=level_boxes,
                    box_indices=box_indices,
                    crop_size=self.interface.POOL_SHAPE,
                    method=self.interface.INTERPOLATION_METHOD
                )
            #tf.print('Pyramid layer:', resized_cropped.shape, output_stream=sys.stdout)
            pooled.append(resized_cropped)
        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0, name="concat_pooled_PyramidROIAlign")

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0, name="concat_box_to_level_PyramidROIAlign")
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1, name="concat_box_to_level2")

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0, name="concat_shape_PyramidROIAlign")
        pooled = tf.reshape(pooled, shape)
        return pooled

    @staticmethod
    @tf.function
    def log2_graph(x):
        """Implementation of Log2. TF doesn't have a native implementation."""
        return tf.math.log(x) / tf.math.log(2.0)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

    def get_config(self):
        config = super().get_config()
        config.update({
            "interface": self.interface.to_dict(),
        })
        return config
