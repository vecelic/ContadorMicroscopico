import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load the model and its configuration
def load_model(model_dir, config_path):
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore the variable global for the current graph
    detection_model.load_weights(tf.compat.v2.train.latest_checkpoint(model_dir))

    return detection_model

# Define a function to load the image and convert it to numpy array
def load_image_into_numpy_array(image_path):
    image_np = np.array(image.imread(image_path))
    return image_np

# Define a function to handle the detection result
def detect_fn(image, detection_model):
    # Prepare the image and the graph
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections = detection_model(input_tensor)

    return detections

# Define a function to visualize the detection results
def visualize_output(image, boxes, classes, scores):
    # Visualize the detection results on the frame
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    # Display the frame with the detection results
   nostic cv2_.modeimshow=False
    ('Object D')
etection   ', cv # Display2. theresize frame with(image the, detection (8 results
00   , cv2 6.0im0)))show
('Object    cv D2ete.waitctionKey',(0 cv2)
.    cvresize(2.image,destroy (AllWindows8()00

,# Set  up the test6 images0
TEST0_IMAGE)))
_PATH    cvS2 = ['.waittest1Key.(0jpg',) '
test2    cv2..jpg']
destroy
AllWindows# Set up()
 the
 model
# Setmodel_ thedir path to = ' thepath/ configuration fileto/ and theyour/ checkmodel/point
directorymodel_dir = ''
path/config_topath =/your 'path/model/to/directory/your'/config
config/file_'path =
d 'path/to/eteyour/ction_modelpipeline = load.config_model'(model

_dir#, config Load the_path model
)
dete
#ction Load the_model category index =
category load__indexmodel( = labelmodel__mapdir,_util.create config__categorypath)_index

_from#_label Define themap(' test imagepath/ paths
to/TESTyour/label_IMAGE_map/PATHSfile') =

 ['path# Set/to up the/your object count dictionary/test/
object_imagecounts/1', ' = {}

path/to# Create a/your/ session to execute thetest/image graph
with/2'] tf.compat

# Define the.v1.Session() as sess:
 category index    for for the image_path dete in TEctions
ST_category_IMAGE_index =PATHS {1:
       : {' image_id':np = 1 load_, 'image_name':into_numpy 'person'},_array( 2image_path: {'id)
':         boxes, classes, scores2, = detect_ 'namefn(': 'image_npbicy, detectioncle'_model}}
)

#        visualize Define a_output dictionary to keep(image_ track of thenp, boxes object counts
object, classes, scores)_counts