import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    right_points_above = []  # list of x values when y = 540 of the right line
    right_points_below = []  # list of x values when y = 320 of the right line
    
    left_points_above = []   # list of x values when y = 540 of the left line
    left_points_below = []   # list of x values when y = 320 of the left line
    
    img_shape = img.shape
    
    for line in lines:        
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope > 0.3: # right line 
                right_points_above.append(x2+(img_shape[0]/2-y2)/slope)
                right_points_below.append(x2+(img_shape[0]-y2)/slope)
            elif slope < -0.3: #left line
                left_points_above.append(x2+(img_shape[0]/2-y2)/slope)
                left_points_below.append(x2+(img_shape[0]-y2)/slope)
                
    right_point_above = (int(np.median(right_points_above)), int(img_shape[0]/2))
    right_point_below = (int(np.median(right_points_below)), img_shape[0])
                
    left_point_above = (int(np.median(left_points_above)), int(img_shape[0]/2))
    left_point_below = (int(np.median(left_points_below)), img_shape[0])
         
    
    cv2.line(img, right_point_above, right_point_below, color, thickness=8) #right_line
    cv2.line(img, left_point_above, left_point_below, color, thickness=8) #left_line



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

    
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

class EdgeFinder:
    def __init__(self, initial_img, filter_size=1, threshold1=0, threshold2=0, rho=1, theta= 1, threshold = 20, min_line_len=10, max_line_gap=5):
        self.image = initial_img

        self.grayimage = cv2.cvtColor(initial_img, cv2.IMREAD_GRAYSCALE)
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._rho = rho
        self._theta = theta
        self._threshold = threshold
        self._min_line_len = min_line_len
        self._max_line_gap = max_line_gap

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()
    
        def onchangeRho(pos):
            self._rho = pos
            self._render()

        def onchangeTheta(pos):
            self._theta = pos
            self._render()

        def onchangeThreshold(pos): # number of points in a line
            self._threshold = pos
            self._render()

        def onchangeMinLineLen(pos):
            self._min_line_len = pos
            self._render()
        
        def onchangeMaxLineGap(pos):
            self._max_line_gap = pos
            self._render()


        # cv2.namedWindow('edges')
        cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('edges', 1200, 1200) 

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges', self._filter_size, 20, onchangeFilterSize)
        cv2.createTrackbar('threshold', 'edges', self._threshold, 255, onchangeThreshold)
        cv2.createTrackbar('rho', 'edges', self._rho, 255, onchangeRho)
        cv2.createTrackbar('theta', 'edges', self._theta, 180, onchangeTheta)
        cv2.createTrackbar('min_line_len', 'edges', self._min_line_len, 200, onchangeMinLineLen)
        cv2.createTrackbar('max_line_gap', 'edges', self._max_line_gap, 200, onchangeMaxLineGap)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def rho(self):
        return self._rho

    def theta(self):
        return self._theta

    def threshold(self):
        return self._threshold

    def min_line_len(self):
        return self._min_line_len

    def max_line_gap(self):
        return self._max_line_gap

    def edgeImage(self):
        return self._edge_img

    def finalImage(self):
        return self._final_img



    def _render(self):
        self._smoothed_img = cv2.GaussianBlur(self.grayimage, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._edge_img = cv2.Canny(self._smoothed_img, self._threshold1, self._threshold2)
        
        imshape = self.grayimage.shape
        vertices = np.array([[(int(imshape[1]/2)+100, int(2*imshape[0]/3)), (int(imshape[1]/2)-100, int(2*imshape[0]/3)), (100, imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
        masked_edges = region_of_interest(self._edge_img, vertices)



        lines = hough_lines(masked_edges, self._rho, self._theta*np.pi/180, self._threshold, self._min_line_len, self._max_line_gap)
        self._final_img = weighted_img(lines, self.image, α=0.8, β=1., γ=0.)
        
        
        cv2.imshow('edges', cv2.cvtColor(self._edge_img, cv2.COLOR_BGR2RGB))
        cv2.imshow('final', cv2.cvtColor(self._final_img, cv2.COLOR_BGR2RGB))
