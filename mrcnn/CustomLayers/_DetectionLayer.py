import tensorflow as tf
from mrcnn.Utils.Interface import Interface
from ._Common import norm_boxes_graph, refine_detections_graph
from mrcnn.Configs import Config
from mrcnn.Utils.utilfunctions import batch_slice, parse_image_meta_graph


class DetectionInterface(Interface):
    def __init__(self, config: Config):
        self.BBOX_STD_DEV = config.BBOX_STD_DEV
        self.DETECTION_MIN_CONFIDENCE = config.DETECTION_MIN_CONFIDENCE
        self.DETECTION_MAX_INSTANCES = config.DETECTION_MAX_INSTANCES
        self.DETECTION_NMS_THRESHOLD = config.DETECTION_NMS_THRESHOLD
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.BATCH_SIZE = config.BATCH_SIZE


class DetectionLayer(tf.keras.layers.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns: [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config: Config, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)

        self.interface = DetectionInterface(config)

        self.refine_detections_graph = refine_detections_graph
        self.norm_boxes_graph = norm_boxes_graph
        self.batch_slice = batch_slice
        self.parse_image_meta_graph = parse_image_meta_graph

    @tf.function
    def call(self, inputs):
        """ The input is a list of params

        Params:
        - rois: ProposalLayer object
        - mrcnn_class: KL.TimeDistributed(KL.Activation("softmax")). Classifier probabilities
        - mrcnn_bbox: KL.Reshape. Deltas to apply to proposal boxes
        - image_meta: KL.Input. Image details

        Returns: A reshaped [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in normalized
            coordinates
        """
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = self.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = self.norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = self.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: self.refine_detections_graph(x, y, w, z, self.interface),
            self.interface.IMAGES_PER_GPU
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.interface.BATCH_SIZE, self.interface.DETECTION_MAX_INSTANCES, 6]
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "interface": self.interface.to_dict(),
        })
        return config

    def compute_output_shape(self, input_shape):
        return (None, self.interface.DETECTION_MAX_INSTANCES, 6)
