import cv2


class StereoDisparity:
    def __init__(self, img_left=None, img_right=None):
        self._img_left = img_left
        self._img_right = img_right

    def compute_disparity(self):
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(self._img_left, self._img_right)
        return cv2.normalize(
            disparity,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    """
    setter
    """

    def set_img_left(self, img_left):
        self._img_left = img_left

    def set_img_right(self, img_right):
        self._img_right = img_right

    """
    Calculate the corresponding coordinates of the right image
    """

    def get_img_right_coresponding_coordinates(self, img_left_x, img_left_y, disparity):
        if disparity[img_left_x, img_left_y] == 0:
            return None

        img_right_x = img_left_x
        img_right_y = img_left_y - disparity[img_left_x, img_left_y]

        return (img_right_x, img_right_y)

    """
    Draw a circle on the image
    """

    def draw_circle_on_img(self, img, x, y, radius, color):
        return cv2.circle(img.copy(), (y, x), radius, color, -1)
