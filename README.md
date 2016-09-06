# Brief description

This project aims to demonstrate and implement the benefits of using parallel processing using NVIDIA CUDA over linear processing.

# Semiglobal Matching (SGM)
Gray scale images will be processed in order to determine the disparity image between two images of a stereo pair. The disparity image assigns each pixel with a value proportional to the distance between that pixel (generally in the left image) and the corresponding pixel in the right image of the stereo pair.

Each image will be modelled as an array of integers which values range from 0 to 255. The values in the image specify the pixel luminance, hence a value of 0 indicates a black pixel and a value of 255 indicates a white pixel. An image will be stored in memory as an array (or matrix) of integer values where each element of the array/matrix corresponds to a pixel in the image.

The left and right images, provided by a stereo pair, will be processed for determining the disparity image using a simplified Semi-Global Matching method [1] with *Birchfield and Tomasi* measure for pixel cost calculation.

It can be assumed that both images have rectified epipolar geometry. This method finds the disparity image in 3 stages: first, it determines the cost of each pixel in the left image for different disparity values (up to a given disparity range) by comparing this image with the right image; then an aggregation of the costs is performed by minimizing an energy constraint on the path of disparities that lead to each point on several directions; finally, from the aggregated cost for each disparity, the disparity image may be determined by finding, for each pixel, the disparity that minimizes the cost.
# Project description
The objective of this work was to start from the linear source, which includes a C implementation of the Semi-Global Matching method (adapted from code available at [2]) and the **testDiffs** tool to compare images, and it was developed improved versions of the Semi-Global-Matching method using the CUDA platform.

Images may be of any size. The function **sgmDevice()** should encapsulate all the operations of preparation, execution and result retrieval of the CUDA kernel(s). Images can be tested using any GPU with *compute capability 1.3*. 
# Running the project
CUDA SDK must be installed and basic development tools.

`./sgm` runs with default of device 0 (if more than one CUDA GPU available), disparity 32 for input of *lbull.pgm* for left image and *rbull.pgm* for right image.

Output is *d_dbull.pgm* for device output and *h_dbull.pgm* for host output.

Calculations are done in both in host computer and device so that a comparison with *testDiffs* can be tested to check if the output is the same and no errors occurred.

# Results

![implementation gains](https://github.com/luminoso/sgm_cuda/raw/master/results_over_implementations.jpg)

This graph shows the advantage of using CUDA in each function compared with running it in the CPU. Time needed to generate same results versus continuous implementation of SGM algorithm.

![disparity gains](https://github.com/luminoso/sgm_cuda/blob/master/results_disp_range.jpg)

This graph shows how CUDA keeps an constant time for calculation over disparity range versus the linear growth using linear computation. Time needed to generate same results versus disparity range.

# Bibliography
[1] Heiko Hirschmüller, Stereo Processing by Semiglobal Matching and Mutual Information, IEEE Trans. Pattern Analysis and Machine Intelligence, 30(2):328–341, 2008

[2] Code to Semi Global Matching. http://lunokhod.org/?p=1403
