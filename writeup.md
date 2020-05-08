# **Finding Lane Lines on the Road** 

## Writeup note

In this project, we implement the algorithm to automate the process of finding landlines on the road. The tools include the color-selection, the region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Transform line detection.


**Finding Lane Lines on the Road**

The goals of this project are the following:
* The pipeline that finds lane lines on the road

* Reflect on your work in a written report



### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipeline include these steps

- Query the image from database
- Transform an image to grayscale
- Apply gaussian_blur to smoothen the image
- Utilize the Canny Edge Detection to find the edges
- Define the vertices of our region_of_interest and apply to the current image
- Draw the lines using the Hough Transformation
- Combine with the original image to make the output

[image1]: ./test_images_output/solidYellowCurve.png "solid Yellow Curve"
[image2]: ./test_images_output/solidWhiteCurve.png "solid White Curve"

After implementing the pipeline, our algorithm can detect some of the lane lines in straightforward cases, however, during the curve or change lane situation there are some issues.


To draw a single line on the left and right lanes, I modified the draw_lines() function by using the slope = $(y2-y1)/(x2-x1)$ of each line to separate the left and right lane lines.

Afterwards, for each case, e.g. right-hand landline, we imagine of two horizontal
lines. One is at the bottom of the image and another is at the position of $y = 2/3$ of the image width. By calculating the all possible intersections of these lines and the two horizontal lines, we can estimate the medians of the points for the final single line.   

[image3]: ./test_images_output/solidYellowLeft.png "solid Yellow Left"
[image4]: ./test_images_output/solidYellowCurve2.png "solid White Curve"


### 2. Identify potential shortcomings with your current pipeline

One issue can happen as in the challenge that the algorithm is not very robust when the car moves through different scenery. Especially, when there are trees and shades on the road, it is more difficult to have good lines after the Hough Transformation.

How to automate the parameters tuning is another challenge for this project. There are 7 of them including
- GaussianBlur Kernel size
- Threshold 1 and 2 (high and low) of the Canny operation
- Rho and Theta of the Hough Lines part
- Point threshold, max_line_gap, and min_line_len

We also include the code for the manual tuning in this repo

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to label the lane line manually for several images. Afterwards, we can train a machine learning model to automate parameter tuning.

Another approach is to improve our draw_lines function by removing the outliers from the list of points for drawing the two lines.
