# Result of a single YOLOv5 bottle detection frame.
std_msgs/Header header
bool detected            # true if a bottle was found in this frame
float32 score            # confidence of the best detection
float32 x_min            # bounding box in original image pixels
float32 y_min
float32 x_max
float32 y_max
uint32 image_width       # width of the source image (pixels)
uint32 image_height      # height of the source image (pixels)
