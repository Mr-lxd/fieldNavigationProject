#include "ImgPro.h"
#include<iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random> 
#include<opencv2/imgproc/types_c.h>
#include <omp.h>


int CImgPro::imgCols = -1, CImgPro::imgRows = -1;

void CImgPro::NormalizedExG(Mat& srcimg, Mat& outimg)
{
	cvtColor(srcimg, outimg, COLOR_RGB2GRAY);		// convert input image to gray image
	unsigned char* in;		// input image pointer
	unsigned char* out;
	unsigned char R, G, B;
	uchar temp1;
	float r, g, b;		// The normalized difference vegetation index

	for (int i = 0; i < srcimg.rows; i++)
	{
		//Obtain the base address of the data located in the i-th row, and retrieve the pointers to the input and output image data of the current row
		//The variable "data" points to the starting address of the image data, variable "step" represents the number of bytes occupied by each row of data
		in = (unsigned char*)(srcimg.data + i * srcimg.step);
		out = (unsigned char*)(outimg.data + i * outimg.step);
		for (int j = 0; j < srcimg.cols; j++)
		{
			//Retrieve the channel value of the j-th pixel in each row.
			B = in[3 * j];
			G = in[3 * j + 1];
			R = in[3 * j + 2];

			float sumRGB = (float)(R + G + B);

			b = (float)B / sumRGB;
			g = (float)G / sumRGB;
			r = (float)R / sumRGB;

			if (2 * g - r - b < 0)
				temp1 = 0;
			else if (2 * g - b - r > 1)
				temp1 = 255;
			else
				temp1 = (2 * g - b - r) * 255;

			out[j] = temp1;


			//The current results of the ExG-ExR method are not satisfactory, particularly in cases where the crops have a significant green component.
			/*float ExG = 2*G - R - B;
			float ExR = 1.4*R - G;
			if (G>R && G>B && ExG - ExR > 0)
			{
					out[j] = ExG - ExR;
			}
			else
			{
				out[j] = 0;
			}*/

		}
	}
}

float CImgPro::euclidean_distance(Point a, Point b)
{
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	return sqrt(pow(dx, 2) + pow(dy, 2));
}

float CImgPro::calculateNonZeroPixelRatio(Mat& img)
{

	int nonZeroPixelCount = cv::countNonZero(img);
	int totalPixelCount = img.rows * img.cols;
	float ratio = (float)nonZeroPixelCount / totalPixelCount;
	return ratio;

}

double CImgPro::thresholdingSigmoid(double NonZeroPixelRatio, double k, double x)
{
	//threshold = 1 / (1 + exp(-k * (NonZeroPixelRatio - x0)))
	double exp_part = exp(-k * (NonZeroPixelRatio - x));

	double numerator = 1;

	double denominator = 1 + exp_part;

	double result = numerator / denominator;

	return result;
}

Point CImgPro::centroid(vector<Point>& points)
{
	float sum_x = 0.0, sum_y = 0.0;
	for (const auto& p : points) {
		sum_x += p.x;
		sum_y += p.y;
	}
	return Point(sum_x / points.size(), sum_y / points.size());
}

int CImgPro::calculate_x(Point p, float k, int outimg_rows)
{
	float b = p.y - k * p.x;

	int x = (outimg_rows - b) / k;

	return x;
}

Point CImgPro::min(vector<Point>& points) const {
	assert(!points.empty());
	return *std::min_element(points.begin(), points.end(),
		[](const Point& a, const Point& b) { return a.x < b.x; });
}

Point CImgPro::max(vector<Point>& points) const {
	assert(!points.empty());
	return *std::max_element(points.begin(), points.end(),
		[](const Point& a, const Point& b) { return a.x < b.x; });
}

int CImgPro::isLeftOrRight(const Point& a, const Point& b, const Point& c)
{
	float side = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
	if (side > 0)
		return -1;		//If the cross product is greater than zero, the point is on the left of the line
	else if (side < 0)
		return 1;		//If the cross product is less than zero, the point is on the right of the line
	/*else if (((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) == 0)
		return 0;*/		//the point is on the the line
}

bool CImgPro::isClusterPassed(const Cluster& cluster, const Point& a, const Point& b, char ID)
{
	int inliers = 0;
	//Point l_min = Point(a.x - 5, a.y + 4), l_max = Point(b.x - 4, b.y + 5);//��
	//Point r_min = Point(a.x + 4, a.y + 5), r_max = Point(b.x + 5, b.y + 4);//��
	Point c_min = Point(a.x - 1, a.y + 12), c_max = Point(b.x + 1, b.y + 12);//��
	//k1 = (float)(l_min.y - a.y) / (l_min.x - a.x), k2 = (float)(l_max.y - b.y) / (l_max.x - b.x), k3 = (float)(r_min.y - a.y) / (r_min.x - a.x);
	//k4 = (float)(r_max.y - b.y) / (r_max.x - b.x), 
	k5 = (float)(c_min.y - a.y) / (c_min.x - a.x), k6 = (float)(c_max.y - b.y) / (c_max.x - b.x);

	//if (ID == 'l') {
	//
	//	for (const Point& p : cluster.points) {
	//		int i = isLeftOrRight(a, l_min, p);
	//		int j = isLeftOrRight(b, l_max, p);
	//		if (i == 1 && j == -1)
	//			inliers++;
	//	}
	//}
	//if (ID == 'r') {
	//	for (const Point& p : cluster.points) {
	//		int i = isLeftOrRight(a, r_min, p);
	//		int j = isLeftOrRight(b, r_max, p);
	//		if (i == 1 && j == -1)
	//			inliers++;
	//	}
	//}
	if (ID == 'c') {
		for (const Point& p : cluster.points) {
			int i = isLeftOrRight(a, c_min, p);
			int j = isLeftOrRight(b, c_max, p);
			if (i == 1 && j == -1)
				inliers++;
		}
	}

	if (inliers != 0) {
		return true;
	}
	else
	{
		return false;
	}
}

Mat CImgPro::MedianBlur(Mat srcimg, int kernel_size)
{
	Mat MedianBlurImg(srcimg.size(), CV_8UC1);
	medianBlur(srcimg, MedianBlurImg, kernel_size);

	return MedianBlurImg;

	//Mat dstimg = srcimg.clone();
	//int numThreads = 4; // 根据你的 CPU 核心数选择线程数
	//int rowsPerThread = srcimg.rows / numThreads;

	//// 创建多个线程，并行处理不同的行段
	//vector<thread> threads;
	//for (int i = 0; i < numThreads; ++i) {
	//	int startRow = i * rowsPerThread;
	//	int endRow = (i == numThreads - 1) ? srcimg.rows : (i + 1) * rowsPerThread;
	//	// 使用 std::bind 来绑定类成员函数和类实例
	//	threads.push_back(thread(std::bind(&CImgPro::applyMedianBlur, this, cref(srcimg), ref(dstimg), startRow, endRow, kernel_size)));
	//}

	//// 等待所有线程完成
	//for (auto& t : threads) {
	//	t.join();
	//}

	//return dstimg;
}

pair<int, int> CImgPro::NZPR_to_Erosion_Dilation(float NZPR, Mat& img)
{
	int Erosion = 0, Dilation = 0;

	if (img.rows == 480 && img.cols == 640)
	{
		//if (NZPR > 0.06 && NZPR <= 0.1) {
		//	Erosion = 1;
		//	Dilation = 0;
		//}
		//if (NZPR > 0.1 && NZPR <= 0.2) {
		//	Erosion = 2;
		//	Dilation = 1;
		//}
		//if (NZPR > 0.2) {
		//	Erosion = 3;
		//	Dilation = 1;
		//}
		if (NZPR > 0.06)
		{
			Erosion = round(-4.0229 * NZPR * NZPR + 11.7543 * NZPR - 0.17);
		}

		Dilation = floor(Erosion / 2.0);
	}

	if (img.rows == 1080 && img.cols == 1920)
	{
		/*if (NZPR > 0.06 && NZPR <= 0.1) {
			Erosion = 1;
			Dilation = 0;
		}
		if (NZPR > 0.1 && NZPR <= 0.2) {
			Erosion = 3;
			Dilation = 2;
		}
		if (NZPR > 0.2 && NZPR <= 0.3) {
			Erosion = 4;
			Dilation = 2;
		}
		if (NZPR > 0.3 && NZPR <= 0.4) {
			Erosion = 5;
			Dilation = 3;
		}
		if (NZPR > 0.4) {
			Erosion = 7;
			Dilation = 4;
		}*/

		if (NZPR >= 0.06 && NZPR < 0.2) {
			Erosion = round(53.5714 * NZPR * NZPR + 3.9286 * NZPR + 0.0714);
		}
		else if (NZPR >= 0.2 && NZPR <= 0.5) {
			Erosion = 10 * NZPR + 2;
		}

		Dilation = floor(Erosion / 2.0);
	}

	if (img.rows == 3072 && img.cols == 4096)
	{
		if (NZPR >= 0.06 && NZPR < 0.2) {
			Erosion = round(16.75 * NZPR * NZPR + 25.81 * NZPR - 0.1);
		}
		else if (NZPR >= 0.2 && NZPR <= 0.5) {
			Erosion = trunc(50 * NZPR * NZPR - 5 * NZPR + 8);
		}
		else {
			Erosion = -1;
		}

		Dilation = floor(Erosion / 2.0);
	}



	return{ Erosion, Dilation };
}

pair<Mat, vector<int>> CImgPro::EightConnectivity(Mat& img, float cof)
{
	Mat labels; // Output labeled image
	int num_labels; // Number of connected components
	Mat stats; // Output statistics for each connected component (bounding box, area, etc.)
	Mat centroids; // Output centroids for each connected component
	num_labels = connectedComponentsWithStats(img, labels, stats, centroids, 8, CV_32S);
	vector<int> firstHistogram(img.cols, 0);

	double sum_area = 0.0;
	double mean_area = 0.0;
	vector<int> area_cache(num_labels); // 缓存每个连通区域的面积

	for (int i = 1; i < num_labels; i++) { // Start from 1 to skip the background
		int area = stats.at<int>(i, CC_STAT_AREA);
		area_cache[i] = area; // 缓存区域面积
		sum_area += area; //  Accumulate area of each connected component
	}
	mean_area = sum_area / (num_labels - 1); // Calculate mean area, excluding background

	Mat output = Mat::zeros(labels.size(), CV_8UC1);
	double mean_area_threshold = mean_area * cof;

	//auto start1 = std::chrono::high_resolution_clock::now();
	//for (int i = 0; i < labels.rows; i++) {
	//	for (int j = 0; j < labels.cols; j++) {
	//		int label = labels.at<int>(i, j); // Get label of the current pixel
	//		// Check if area of the connected component containing the current pixel is greater than or equal to the mean area
	//		if (label > 0 && area_cache[label] >= mean_area_threshold) {
	//			output.at<uchar>(i, j) = 255;
	//			if (j >= 0.325 * labels.cols && j <= 0.675 * labels.cols) {//// Process region where the image does not follow normal distribution, like: 112212
	//				firstHistogram[j]++;
	//			}

	//		}
	//	}
	//}
	//auto end1 = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed1 = end1 - start1;


	// ptr is more fast than at
	int* labels_ptr = labels.ptr<int>();
	uchar* output_ptr = output.ptr<uchar>();
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels_ptr[i * labels.cols + j]; // Get label of the current pixel
			// Check if area of the connected component containing the current pixel is greater than or equal to the mean area
			if (label > 0 && area_cache[label] >= mean_area_threshold) {
				output_ptr[i * output.cols + j] = 255;
				if (j >= 0.325 * labels.cols && j <= 0.675 * labels.cols) {
					firstHistogram[j]++;
				}
			}
		}
	}

	return { output,firstHistogram };
}

void CImgPro::processImageWithWindow(Mat& srcimg, Mat& outimg, Cluster& points, int windowWidth, int windowHeight, int flag)
{
	int rows = srcimg.rows;
	int cols = srcimg.cols;

	// Pointer-based access to image data
	uchar* src_data = srcimg.ptr<uchar>();
	uchar* out_data = outimg.ptr<uchar>();

	// Check if the input image is single channel or 3 channels
	bool is_single_channel = (srcimg.channels() == 1);
	bool is_three_channel = (srcimg.channels() == 3);

	//auto start1 = std::chrono::high_resolution_clock::now();
	//for (int y = 0; y <= rows - windowHeight; y += windowHeight)
	//{
	//	for (int x = 0; x <= cols - windowWidth; x += windowWidth)
	//	{
	//		// Calculate the average value of pixel coordinates within the window
	//		int count = 0;
	//		int sumX = 0.0, sumY = 0.0;
	//		float avgX = 0.0, avgY = 0.0;
	//		for (int wy = 0; wy < windowHeight; ++wy)
	//		{
	//			for (int wx = 0; wx < windowWidth; ++wx)
	//			{
	//				if (srcimg.channels() == 1) {
	//					if (srcimg.at<uchar>(y + wy, x + wx) != 0)
	//					{
	//						sumY += y + wy;
	//						sumX += x + wx;
	//						count++;
	//						//img.at<uchar>(y + wy, x + wx) = 0;
	//					}
	//				}
	//				if (srcimg.channels() == 3) {
	//					if (srcimg.at<Vec3b>(y + wy, x + wx) != Vec3b(0, 0, 0))
	//					{
	//						sumY += y + wy;
	//						sumX += x + wx;
	//						count++;
	//					}
	//				}
	//			}
	//		}

	//		if (sumX != 0 || sumY != 0)
	//		{
	//			avgX = (float)sumX / count;
	//			avgY = (float)sumY / count;
	//			if (flag == 1) {
	//				outimg.at<uchar>(avgY, avgX) = 255;

	//			}
	//			if (flag == 2)
	//			{
	//				outimg.at<Vec3b>(avgY, avgX) = Vec3b(0, 0, 0);
	//			}

	//			points.points.push_back(Point(avgX, avgY));
	//		}
	//	}
	//}
	//auto end1 = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed1 = end1 - start1;


	// Iterate over windows in the image
	for (int y = 0; y <= rows - windowHeight; y += windowHeight)
	{
		for (int x = 0; x <= cols - windowWidth; x += windowWidth)
		{
			// Calculate the sum of coordinates inside the window
			int count = 0;
			int sumX = 0, sumY = 0;
			for (int wy = 0; wy < windowHeight; ++wy)
			{
				for (int wx = 0; wx < windowWidth; ++wx)
				{
					int idx = (y + wy) * cols + (x + wx);  // Index into the image data
					if (is_single_channel)
					{
						if (src_data[idx] != 0)  // If pixel is non-zero in a single-channel image
						{
							sumY += (y + wy);
							sumX += (x + wx);
							count++;
						}
					}
					else if (is_three_channel)
					{
						Vec3b* pixel = (Vec3b*)(src_data + idx * 3);
						if (pixel->val[0] != 0 || pixel->val[1] != 0 || pixel->val[2] != 0)  // If pixel is non-zero in a 3-channel image
						{
							sumY += (y + wy);
							sumX += (x + wx);
							count++;
						}
					}
				}
			}

			// If the window has non-zero pixels, calculate the average coordinates
			if (count > 0)
			{
				float avgX = (float)sumX / count;
				float avgY = (float)sumY / count;

				// Set the value of the output image based on the flag
				if (flag == 1)
				{
					// Assuming outimg is a single-channel image
					out_data[(int(avgY) * cols) + int(avgX)] = 255;
				}
				else if (flag == 2)
				{
					// Assuming outimg is a 3-channel image
					Vec3b* pixel = (Vec3b*)(out_data + (int(avgY) * cols + int(avgX)) * 3);
					*pixel = Vec3b(0, 0, 0);  // Set to black
				}

				// Store the point coordinates in the cluster
				points.points.push_back(Point(avgX, avgY));
			}
		}
	}
}

void CImgPro::verticalProjection(Mat& img, const vector<Cluster>& clusters, double cof)
{
	vector<int> histogram(img.cols, 0);

	for (auto& c : clusters) {
		for (auto& p : c.points) {
			histogram[p.x]++;
		}
	}

	// Find the maximum value in the histogram
	int y_max = *std::max_element(histogram.begin(), histogram.end());

	//Size size(img.cols, 1.2 * y_max);
	//Mat histogramImg(size, CV_8UC1, Scalar(0));
	//for (int i = 0; i < histogram.size(); i++) {
	//	// Draw the histogram line
	//	line(histogramImg, Point(i, 1.2 * y_max), Point(i, 1.2 * y_max - histogram[i]), Scalar(255), 1);
	//}

	// Draw a horizontal line for thresholding
	int horizontal_line_height = cof * y_max;
	//line(histogramImg, Point(0, horizontal_line_height), Point(histogramImg.cols - 1, horizontal_line_height), Scalar(255), 1);

	// Find the x-coordinate of the intersection between the histogram and the horizontal line
	/*bool flag = true;
	x_max = -1, x_min = histogramImg.cols + 1;
	for (int i = 0; i < histogram.size(); i++) {
		if (histogramImg.at<uchar>(horizontal_line_height, i) == 255) {
			if (flag) {
				x_min = i;
				flag = false;
			}
			if (i > x_max) {
				x_max = i;
			}
		}
	}*/

	x_min = -1, x_max = -1;

	//Scan the histogram to find the min and max x coordinates that cross the threshold
	bool flag = true;
	for (int i = 0; i < histogram.size(); i++) {
		if (histogram[i] >= 1.2 * y_max - horizontal_line_height) {
			if (flag) {
				x_min = i; // First occurrence of the threshold crossing
				flag = false;
			}
			x_max = i; // Update max x value as we go through the histogram
		}
	}

	//return histogramImg;
}

int CImgPro::verticalProjectionForCenterX(const vector<int>& histogram)
{
	int y_max = -1, centerX = -1;
	for (int i = 0; i < histogram.size(); i++)
	{
		if (histogram[i] > y_max) {
			y_max = histogram[i];
			centerX = i;
		}
	}


	//Size size(histogram.size(), 1.2 * y_max);
	//Mat histogramImg(size, CV_8UC1, Scalar(0));
	//for (int i = 0; i < histogram.size(); i++) {
	//	// Draw the histogram line
	//	line(histogramImg, Point(i, 1.2 * y_max), Point(i, 1.2 * y_max - histogram[i]), Scalar(255), 1);
	//}


	//return { histogramImg, centerX };

	return centerX;
}

void CImgPro::retainMainStem(vector<Cluster>& clusters)
{
	for (auto& c : clusters) {
		auto it = c.points.begin();
		while (it != c.points.end()) {
			if (it->x < x_min || it->x > x_max) {
				// Remove points outside the desired range
				it = c.points.erase(it);
			}
			else {
				// Keep points inside the desired range
				++it;
			}
		}
	}

}

pair<Mat, float> CImgPro::OTSU(Mat src)
{
	float NonZeroPixelRatio = 0.0f;

	Mat OtsuImg;
	double thresh = cv::threshold(src, OtsuImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//binary
	//Mat OtsuImgV1(src.size(), CV_8UC1, Scalar(0));
	int nonZeroPixelCount = 0;

	//auto start16 = std::chrono::high_resolution_clock::now();
	//for (int i = 0; i < src.rows; i++)
	//{
	//	for (int j = 0; j < src.cols; j++)
	//	{
	//		//Foreground
	//		if (src.at<uchar>(i, j) > 0.8 * thresh)//Lower the threshold to prevent filtering out darker green tones.
	//		{
	//			OtsuImgV1.at<uchar>(i, j) = 255;
	//			nonZeroPixelCount++;
	//		}
	//		//background
	//		//else
	//		//{
	//		//	OtsuImg.at<uchar>(i, j) = 0;
	//		//}
	//	}
	//}
	//auto end16 = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> elapsed16 = end16 - start16;

	uchar* pSrc = src.data;
	uchar* pOtsuImg = OtsuImg.data;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int idx = i * src.cols + j;  // Flatten 2D index to 1D
			if (pSrc[idx] > 0.8 * thresh) {  // Check if pixel is foreground
				pOtsuImg[idx] = 255;  // Mark as foreground
				nonZeroPixelCount++;
			}
			else {
				pOtsuImg[idx] = 0;  // Mark as background
			}
		}
	}

	// NonZeroPixelRatio for MorphologicalOperation
	int totalPixel = src.rows * src.cols;
	NonZeroPixelRatio = (float)nonZeroPixelCount / totalPixel;

	return { OtsuImg, NonZeroPixelRatio };
}

Mat CImgPro::MorphologicalOperation(Mat src, int kernel_size, int cycle_num_e, int cycle_num_d)
{
	Mat kernel;

	kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));

	erode(src, src, kernel, Point(-1, -1), cycle_num_e);
	dilate(src, src, kernel, Point(-1, -1), cycle_num_d);

	return src;
}

Mat CImgPro::ClusterPointsDrawing(Mat& src, vector<Cluster>& points)
{
	Mat outimg(src.size(), CV_8UC3, Scalar(0, 0, 0));

	vector<Scalar> colors = {
	Scalar(0, 0, 255),     // red
	Scalar(0, 255, 0),     // green
	Scalar(255, 0, 0),     // blue
	Scalar(255, 255, 0),   // yellow
	Scalar(0, 255, 255),   // cyan
	Scalar(255, 0, 255),   // magenta
	Scalar(128, 0, 0),     // deep red
	Scalar(0, 128, 0),     // dark green
	Scalar(0, 0, 128),      // dark blue
	Scalar(128, 128, 0),   // olive green
	Scalar(128, 0, 128),   // purple
	Scalar(0, 128, 128),   // turquoise
	Scalar(0, 165, 255),   // orange
	};

	// Iterate through each cluster and draw points
	for (int i = 0; i < points.size(); i++) {
		// Determine color for the cluster based on index
		Scalar clusterColor = colors[i % colors.size()];  // Rotate through colors if more than 12 clusters

		for (const auto& p : points[i].points) {
			// Plot the point using the determined color
			circle(outimg, Point(p.x, p.y), 1, clusterColor, -1);
		}
	}


	// Iterate through each set of coordinates and plot them on the image.
	//for (int i = 0; i < points.size(); i++) {
	//	bool firstPoint = true;
	//	int id = i + 1;
	//	for (int j = 0; j < points[i].points.size(); j++) {
	//		int x = points[i].points[j].x;
	//		int y = points[i].points[j].y;

	//		// Determine the brush color based on the group to which the point belongs.
	//		Scalar color = colors[i % colors.size()];
	//		/*Scalar color = colors[i];*/

	//		// Plot the point
	//		circle(outimg, Point(x, y), 1, color, -1);

	//		//Display the group number next to the first point.
	//		//if (firstPoint)
	//		//{
	//		//	Point textPt1(x + 20, y + 90);
	//		//	putText(outimg, to_string(id), textPt1, FONT_HERSHEY_SIMPLEX, 1, color, 3);
	//		//	firstPoint = false;

	//		//	//d1,d2,d3....
	//		//	/*Point textPt2((points[i].averageX + 2029)/2, points[i].averageY-10);
	//		//	string s = "d" + to_string(id);
	//		//	putText(outimg, s, textPt2, FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 8);
	//		//	firstPoint = false;*/
	//		//}

	//		/*arrowedLine(outimg, Point(points[i].averageX, points[i].averageY), Point(2029, points[i].averageY), Scalar(255, 255, 255), 2, 8, 0, 0.1);
	//		arrowedLine(outimg, Point(2029, points[i].averageY), Point(points[i].averageX, points[i].averageY), Scalar(255, 255, 255), 2, 8, 0, 0.1);*/
	//	}

	//}
	/*line(outimg, Point(1316, 0), Point(1316, outimg.rows), Scalar(255, 255, 255), 4, 8, 0);
	line(outimg, Point(2029, 0), Point(2029, outimg.rows), Scalar(255, 255, 255), 4, 8, 0);
	line(outimg, Point(2742, 0), Point(2742, outimg.rows), Scalar(255, 255, 255), 4, 8, 0);
	arrowedLine(outimg, Point(1316, 3000), Point(2742, 3000), Scalar(255, 255, 255), 2, 8, 0, 0.1);
	arrowedLine(outimg, Point(2742, 3000), Point(1316, 3000), Scalar(255, 255, 255), 2, 8, 0, 0.1);*/

	//ROI
	/*k5 = -12, k6 = 12;
	int x1 = calculate_x(points[0].CategID[0], k5, outimg.rows);
	line(outimg, points[0].CategID[0], Point(x1, outimg.rows), Scalar(255, 255, 255), 2, 8, 0);
	int x2 = calculate_x(points[0].CategID[1], k6, outimg.rows);
	line(outimg, points[0].CategID[1], Point(x2, outimg.rows), Scalar(255, 255, 255), 2, 8, 0);*/

	return outimg;
}

void CImgPro::RANSAC(Cluster& cluster, float thresh, Mat& outimg)
{
	//vector<float> dis;
	vector<CImgPro::Cluster> inliers;
	Cluster tempPoints;
	struct Line {
		double slope, intercept;
	};

	random_device rd;		//Local true random number generator
	mt19937 gen(rd());		//Pseudo-random number generator using rd as seed

	int best_inliers = 0;
	float bestSlope = 0.0, bestIntercept = 0.0;
	float iterations = 0.0, ConfidenceLevel = 0.99, Probability = 2.0 / cluster.points.size();

	iterations = log(1 - ConfidenceLevel) / log(1 - pow(Probability, 2));

	for (int j = 0; j < iterations; j++)		//Continuously iterate in this cluster
	{
		// Randomly select two different points
		uniform_int_distribution<> distrib(0, cluster.points.size() - 1);		//Integer distribution object 'distrib' with range [0, n]
		int index1 = distrib(gen);
		int index2 = distrib(gen);
		// Prevent selecting the same index
		while (index2 == index1)
		{
			index2 = distrib(gen);
		}
		Point p1 = cluster.points[index1];
		Point p2 = cluster.points[index2];

		float slope = 0, intercept = 0;
		if (p1.x == p2.x)
		{
			slope = 9999999;
		}
		else {
			slope = (float)(p2.y - p1.y) / (p2.x - p1.x);
		}
		intercept = p1.y - slope * p1.x;
		Line l = { slope, intercept };

		// Calculate distances from all points in this cluster to the line excluding points p1 and p2
		float distance = 0;
		for (const auto& p : cluster.points)
		{
			if (p != p1 && p != p2)
			{
				distance = abs(p.y - l.slope * p.x - l.intercept) / sqrt(1 + l.intercept * l.intercept);
				//Check inliers
				if (distance < thresh)
				{
					tempPoints.points.push_back(p);
					//dis.push_back(distance);
				}
			}
		}


		if (tempPoints.points.size() > best_inliers)
		{
			inliers.clear();
			inliers.push_back(tempPoints);
			best_inliers = tempPoints.points.size();
			bestSlope = l.slope;
			bestIntercept = l.intercept;

			//Update inlier ratio and iterations 
			Probability = (float)best_inliers / cluster.points.size(); // Calculate inlier ratio based on the current best hypothesis inliers
			iterations = log((1 - ConfidenceLevel)) / log((1 - pow(Probability, 2)));
			iterations = 300 * iterations;
			j = 0; //Reset iteration counter
		}


		tempPoints.points.clear();
		//dis.clear();		
	}

	//outimg = ClusterPointsDrawing(outimg, inliers);

	/*Scalar color = CV_RGB(255, 0, 0);
	line(outimg, Point(-bestIntercept / bestSlope, 0), Point((outimg.rows - bestIntercept) / bestSlope, outimg.rows), color, 10, 8, 0);*/

	//perform least-squares fitting on the point set with the most inliers
	leastSquaresFit_edit(inliers[0], outimg);
	//Hough_Line(inliers, outimg);
}

vector<CImgPro::Cluster> CImgPro::KDTreeAcceleratedDBSCAN(Cluster& points, float epsilon, int minPts)
{
	vector<int> clusterIDs(points.points.size(), -1); // Initialize all points as unclassified (-1)
	int currentClusterID = 0;
	unordered_map<int, vector<int>> cachedNeighbours; // 用于缓存邻居查询结果

	buildKDTree(points, epsilon, cachedNeighbours);  // 使用 KD-tree 进行邻居查找

	
	for (int i = 0; i < points.points.size(); ++i) {
		if (clusterIDs[i] == -1) { // if point unclassified
			vector<int> neighbours = cachedNeighbours[i]; // 直接获取缓存的邻居信息

			if (neighbours.size() >= minPts) { // if core point
				expandCluster(points, clusterIDs, currentClusterID, i, epsilon, neighbours, cachedNeighbours);
				currentClusterID++;
			}
		}
	}
	


	// 构建簇
	vector<Cluster> cluster_points(currentClusterID);
	for (int i = 0; i < points.points.size(); ++i) {
		int clusterID = clusterIDs[i];
		if (clusterID != -1) { // 将非噪声点加入对应的簇
			cluster_points[clusterID].points.push_back(Point(points.points[i].x, points.points[i].y));
		}
	}


	// 计算每个簇的质心
	for (Cluster& cluster : cluster_points) {
		if (!cluster.points.empty()) {
			float sumX = 0, sumY = 0;
			for (Point& point : cluster.points) {
				sumX += point.x;
				sumY += point.y;
			}
			cluster.averageX = sumX / cluster.points.size();
			cluster.averageY = sumY / cluster.points.size();
		}
	}

	return cluster_points;
}

vector<CImgPro::Cluster> CImgPro::secondClusterBaseOnCenterX(vector<Cluster>& cluster_points, int imgCenterX, float cof)
{
	//imageCenterX is the baseline

	vector<float> centroidDistances(cluster_points.size(), 0.0f);
	vector<float> centroidXCoords(cluster_points.size(), 0.0f);

	float totalDistance = 0.0f;
	for (int i = 0; i < cluster_points.size(); ++i) {
		centroidDistances[i] = abs(cluster_points[i].averageX - imgCenterX);
		centroidXCoords[i] = cluster_points[i].averageX;
		totalDistance += centroidDistances[i];
	}


	float averageDistance = totalDistance / centroidDistances.size();


	vector<Cluster> center;
	float cof_avgDist = cof * averageDistance;
	for (int i = 0; i < cluster_points.size(); ++i) {
		if (centroidDistances[i] <= cof_avgDist) {
			center.push_back(cluster_points[i]);
		}
	}


	/*    Irrelevant code, kept for future research convenience.   */
	// lfet side
	/*
	while (!left.empty())
	{
		Point minPoint = min(left[0].points);
		Point maxPoint = max(left[0].points);

		temp.CategID.push_back(minPoint);
		temp.CategID.push_back(maxPoint);

		temp.points.insert(temp.points.end(), left[0].points.begin(), left[0].points.end());
		�ƶ�λ�ò�ɾ���þ���
		rotate(left.begin(), left.begin() + 1, left.end());
		left.pop_back();
		temp.ID = 'l';

		for (auto it = left.begin(); it != left.end();) {
			Cluster& cluster = *it;
			bool flag = isClusterPassed(cluster, minPoint, maxPoint, cluster.ID);
			if (flag) {
				temp.points.insert(temp.points.end(), cluster.points.begin(), cluster.points.end());
				it = left.erase(it);
			}
			else {
				++it;
			}
		}

		final_cluster_points.push_back(temp);
		temp.points.clear();
		temp.CategID.clear();
	}
	*/

	Cluster temp;
	vector<Cluster> final_cluster_points;
	Point minPoint = Point(imgCenterX - 0.07 * imgCols, 0); // left boundary
	Point maxPoint = Point(imgCenterX + 0.07 * imgCols, 0);

	temp.CategID.push_back(minPoint);
	temp.CategID.push_back(maxPoint);
	temp.ID = 'c';

	for (auto it = center.begin(); it != center.end();) {
		Cluster& cluster = *it;
		cluster.ID = 'c';
		bool flag = isClusterPassed(cluster, minPoint, maxPoint, cluster.ID);

		if (flag) {
			temp.points.insert(temp.points.end(), cluster.points.begin(), cluster.points.end());
			it = center.erase(it); // Delete visited clusters and update the iterator.
		}
		else {
			++it; // Continue iterating to the next cluster.
		}
	}

	final_cluster_points.push_back(temp);

	//vector<int> firstHistogram(imgCols, 0);
	//for (Cluster& cluster : final_cluster_points) {
	//	for (Point& point : cluster.points) {
	//		int key = point.x;
	//		firstHistogram[key]++;
	//	}
	//}


	temp.points.clear();
	temp.CategID.clear();


	return final_cluster_points;
}

void CImgPro::buildKDTree(Cluster& points, float epsilon, unordered_map<int, vector<int>>& cachedNeighbours) {
	// 创建一个用于存储所有点的 cv::Mat
	cv::Mat data(points.points.size(), 2, CV_32F);  // 2 表示每个点有两个坐标 (x, y)

	// 将每个点的坐标转换为 Mat 格式
	for (int i = 0; i < points.points.size(); ++i) {
		data.at<float>(i, 0) = static_cast<float>(points.points[i].x);
		data.at<float>(i, 1) = static_cast<float>(points.points[i].y);
	}

	// 使用 KD-tree 进行邻居查找
	cv::flann::KDTreeIndexParams indexParams;
	cv::flann::Index kdtree(data, indexParams);  // 构建 KD-tree
 

	int chunk_size;
	// 遍历每个点，查询邻居
	if (CImgPro::imgRows == 480)
	{
		chunk_size = 100;
	}
	else if (CImgPro::imgRows == 1080) // for 1080p
	{
		chunk_size = 1000;
	}
	else
	{
		chunk_size = 2000;
	}

	#pragma omp parallel for schedule(dynamic, chunk_size) 
	for (int i = 0; i < points.points.size(); ++i) {
		std::vector<float> query = { static_cast<float>(points.points[i].x), static_cast<float>(points.points[i].y) };
		std::vector<int> neighbours(20);   // 初始化大小为 20
		std::vector<float> dists(20);      // 初始化大小为 20
		

		// 使用 KD-tree 进行邻居查找，传递正确的 SearchParams 对象
		int foundNeighbours = kdtree.radiusSearch(query, neighbours, dists, epsilon, neighbours.size(), cv::flann::SearchParams(-1, 0.0f, false));

		// 调整 neighbours 和 dists 容器的大小，只保留实际找到的邻居
		if (foundNeighbours > 1) {
			neighbours.resize(foundNeighbours);  // 调整邻居向量大小为实际找到的邻居数
			dists.resize(foundNeighbours);      // 调整距离向量大小
		}
		//else {
		//	neighbours.clear();  // 如果没有找到邻居，则清空该向量
		//}

		// 确保并行修改缓存时是线程安全的
		#pragma omp critical
		{
			cachedNeighbours[i] = neighbours;
		}
	}

}

void CImgPro::expandCluster(Cluster& points, vector<int>& clusterIDs, int currentClusterID,
	int pointIndex, double epsilon, const vector<int>& neighbours, unordered_map<int, vector<int>>& cachedNeighbours)
{
	std::queue<int> processQueue;
	processQueue.push(pointIndex);
	clusterIDs[pointIndex] = currentClusterID; // Mark the current point as the current cluster ID

	while (!processQueue.empty()) {
		int currentIndex = processQueue.front();
		processQueue.pop();

		// 直接从缓存中获取邻居
		vector<int>& newNeighbours = cachedNeighbours[currentIndex];

		if (newNeighbours.size() > 1) { // Number of samples in the neighborhood of density-connected points is greater than or equal to minPts
			for (int i : newNeighbours) {
				if (clusterIDs[i] == -1) { // Unclassified
					clusterIDs[i] = currentClusterID; // Mark density-connected points as the current cluster ID
					processQueue.push(i); // Add newly found neighboring points to the queue
				}
			}
		}
	}

}

//Improved Least Squares Fitting Capable of Fitting Vertical Lines
void CImgPro::leastSquaresFit_edit(Cluster& cluster, Mat& outimg)
{
	//ax+by+c=0
	//for (int i = 0; i < points.size(); i++)
	//{
	double sumX = 0.0, sumY = 0.0, avgX = 0.0, avgY = 0.0;
	for (auto& p : cluster.points)
	{
		sumX += p.x;
		sumY += p.y;
	}
	avgX = sumX / cluster.points.size();
	avgY = sumY / cluster.points.size();

	double L_xx = 0.0, L_yy = 0.0, L_xy = 0.0;
	for (auto& p : cluster.points)
	{
		L_xx += (p.x - avgX) * (p.x - avgX);
		L_xy += (p.x - avgX) * (p.y - avgY);
		L_yy += (p.y - avgY) * (p.y - avgY);
	}

	double m = L_xx + L_yy;
	double n = L_xx * L_yy - L_xy * L_xy;
	double lamd1 = (m + sqrt(m * m - 4 * n)) / 2;
	double lamd2 = (m - sqrt(m * m - 4 * n)) / 2;
	double lamd = std::min(lamd1, lamd2);
	double d = sqrt((L_xx - lamd) * (L_xx - lamd) + L_xy * L_xy);

	double a, b, c;
	if (abs(d) < 1e-6)
	{
		a = 1;
		b = 0;
		c = -a * avgX - b * avgY;
	}
	else
	{
		if (lamd >= L_xx)
		{
			a = L_xy / d;
			b = (lamd - L_xx) / d;
			
		}
		else
		{
			a = -L_xy / d;
			b = (L_xx - lamd) / d;
			
		}
		c = -a * avgX - b * avgY;
	}


	if (b != 0 && (abs(a / b) >= 1))
	{
		Scalar color = CV_RGB(255, 0, 0);
		line(outimg, Point(-c / a, 0), Point((outimg.rows * (-b) - c) / a, outimg.rows), color, 20, 8, 0);
	}

	float firstSlope = (float)-a / b;
	//}
}

void CImgPro::SaveImg(Mat& img, string& inputFilePath)
{
	// 获取用户桌面路径
	char* homePath = std::getenv("USERPROFILE");  // 获取用户主目录
	if (!homePath) {
		std::cerr << "无法获取用户主目录路径。" << std::endl;
		return;
	}
	std::string desktopPath = std::string(homePath) + "\\Desktop\\";  // 拼接到桌面路径

	// 提取输入文件名（不带路径）
	std::string inputFileName = inputFilePath.substr(inputFilePath.find_last_of("/\\") + 1);

	// 设置带有路径的输出文件名
	std::string outfilename = desktopPath + inputFileName;

	// 保存图片
	bool success = cv::imwrite(outfilename, img);
	if (success) {
		std::cout << "图像已保存到: " << outfilename << std::endl;
	}
	else {
		std::cerr << "保存图像失败。" << std::endl;
	}
}

void CImgPro::saveProcessingTimes(int newTime, const std::string& filename)
{
	std::vector<int> times;

	// Step 1: 读取已有的处理时间数据（如果存在）
	std::ifstream infile(filename);
	if (infile.is_open()) {
		std::string line;
		while (std::getline(infile, line)) {
			// 跳过分隔符和 "平均时间" 行
			if (line.find("--END OF BATCH--") != std::string::npos || line.find("平均时间") != std::string::npos) {
				continue;  // 跳过这行
			}

			// 分割每个处理时间数据
			std::stringstream ss(line);
			std::string item;
			while (std::getline(ss, item, '+')) {
				int time = std::stoi(item);  // 将字符串转换为整数
				times.push_back(time);
			}
		}
		infile.close();
	}

	// Step 2: 追加新的处理时间
	times.push_back(newTime);

	// Step 3: 打开文件以追加方式写入
	std::ofstream outfile(filename, std::ios::app);  // 使用 std::ios::app 以追加模式打开文件
	if (!outfile.is_open()) {
		std::cerr << "无法打开文件进行写入: " << filename << std::endl;
		return;
	}

	// Step 4: 记录新添加的时间
	outfile << newTime;

	// 添加 "+" 号，除非是20个数据的结束
	if (times.size() % 20 != 0) {
		outfile << "+";
	}

	// Step 5: 如果已经完成了20张图片，计算并写入平均时间
	if (times.size() % 20 == 0) {
		int startIdx = times.size() - 20;  // 只取最近20个数据的起始位置
		int sum = std::accumulate(times.begin() + startIdx, times.end(), 0);
		double average = static_cast<double>(sum) / 20;

		// 插入分隔符和平均时间
		outfile << "\n--END OF BATCH--\n";
		outfile << "\n平均时间: " << average << " ms\n";
	}

	outfile.close();  // 关闭文件
}

