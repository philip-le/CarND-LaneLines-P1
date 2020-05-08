"""
How to run:
python find_edges.py <image path>
"""

import argparse
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from guiutils import EdgeFinder


def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    args = parser.parse_args()
    file_list = os.listdir(args.filename)

    for image_name in file_list:
        print(f'****************{image_name}*******************')
        initial_img = mpimg.imread(args.filename + image_name)

        edge_finder = EdgeFinder(initial_img, filter_size=13, threshold1=28, threshold2=115)

        print("Edge parameters:")
        print("GaussianBlur Filter Size: %f" % edge_finder.filterSize())
        print("Threshold1: %f" % edge_finder.threshold1())
        print("Threshold2: %f" % edge_finder.threshold2())
        print("Rho: %f" % edge_finder.rho())
        print("Theta: %f" % edge_finder.theta())
        print("Points Threshold: %f" % edge_finder.threshold())
        print("max_line_gap: %f" % edge_finder.max_line_gap())
        print("min_line_len: %f" % edge_finder.min_line_len())


        (head, tail) = (args.filename, image_name)

        (root, ext) = os.path.splitext(tail)

        final_filename = os.path.join("output_images", root + "-final" + ext)
        edge_filename = os.path.join("output_images", root + "-edges" + ext)
        
        cv2.imwrite(edge_filename, edge_finder.edgeImage())
        cv2.imwrite(final_filename, edge_finder.finalImage())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
