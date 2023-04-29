import tensorflow as tf
from ._Common import norm_boxes_graph, refine_detections_graph
from resources.utils import batch_slice, parse_image_meta_graph


class DetectionLayer(tf.keras.layers.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(
            self,
            bbox_std_dev,
            detection_min_confidence,
            detection_max_instances,
            detections_nms_threshold,
            images_per_gpu,
            **kwargs
    ):
        super(DetectionLayer, self).__init__(**kwargs)
        self.bbox_std_dev = bbox_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_max_instances = detection_max_instances
        self.detections_nms_threshold = detections_nms_threshold
        self.images_per_gpu = images_per_gpu

        self.refine_detections_graph = refine_detections_graph
        self.norm_boxes_graph = norm_boxes_graph
        self.batch_slice = batch_slice
        self.parse_image_meta_graph = parse_image_meta_graph

    @tf.function
    def call(self, inputs):
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
            lambda x, y, w, z: self.refine_detections_graph(
                x, y, w, z,
                self.bbox_std_dev,
                self.detection_min_confidence,
                self.detection_max_instances,
                self.detections_nms_threshold
            ), self.images_per_gpu
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.batch_size, self.detection_max_instances, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.detection_max_instances, 6)
