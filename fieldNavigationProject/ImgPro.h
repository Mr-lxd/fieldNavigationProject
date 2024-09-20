#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <opencv2/flann.hpp>




using namespace std;
using namespace cv;


class CImgPro
{
public:
	static int imgCols, imgRows;

	typedef struct
	{
		vector<Point> points;
		vector<Point> CategID;
		double averageX;
		double averageY;
		double X;
		int count;
		char ID;
		char state;
	}Cluster;

	double thresholdingSigmoid(double NonZeroPixelRatio, double k, double centerx);
	Mat MedianBlur(Mat srcimg, int kernel_size);
	Mat verticalProjection(Mat& img, const vector<Cluster>& clusters, double cof);
	Mat MorphologicalOperation(Mat src, int kernel_size, int cycle_num_e, int cycle_num_d);
	Mat ClusterPointsDrawing(Mat& src, vector<Cluster>& points);
	Mat ClusterPoints(Mat& src, vector<Cluster>& points);
	pair<Mat, int> verticalProjectionForCenterX(const vector<int>& histogram);
	pair<Mat, vector<int>> EightConnectivity(Mat& img, float cof);
	pair<Mat, float> OTSU(Mat src);
	pair<int, int>NZPR_to_Erosion_Dilation(float NZPR, Mat& img);
	vector<Cluster> KDTreeAcceleratedDBSCAN(Cluster& points, float epsilon, int minPts);
	vector<Cluster> secondClusterBaseOnCenterX(vector<Cluster>& cluster_points, int imgCenterX, float cof);
	void processImageWithWindow(Mat& srcimg, Mat& outimg, Cluster& points, int windowWidth, int windowHeight, int flag);
	void retainMainStem(vector<Cluster>& clusters);
	void NormalizedExG(Mat& srcimg, Mat& outimg);
	void RANSAC(Cluster& points, float thresh, Mat& outimg);
	void leastSquaresFit_edit(Cluster& cluster, Mat& outimg);
	void SaveImg(Mat& img);
	void saveProcessingTimes(int newTime, const std::string& filename);

private:

	float k1, k2, k3, k4, k5, k6;		//The slopes of the lines corresponding to the minimum and maximum points for the left, center, and right clusters
	float x_max, x_min;		//the x-coordinate of the intersection between the histogram and the horizontal line

	float euclidean_distance(Point a, Point b);
	float calculateNonZeroPixelRatio(Mat& img);
	int calculate_x(Point p, float k, int outimg_rows);
	bool isClusterPassed(const Cluster& cluster, const Point& minPoint, const Point& maxPoint, char ID);
	int isLeftOrRight(const Point& a, const Point& b, const Point& c);
	Point centroid(vector<Point>& points);
	Point min(vector<Point>& points) const;
	Point max(vector<Point>& points) const;
	vector<Cluster> ComparePoints(vector<Cluster>& points);
	vector<int> regionQuery(Cluster& points, Point& point, double epsilon);
	void expandCluster(Cluster& points, vector<int>& clusterIDs, int currentClusterID,
		int pointIndex, double epsilon, const vector<int>& neighbours, unordered_map<int, vector<int>>& cachedNeighbours);
	vector<Cluster> BaseCluster(Mat featureimage, int beginHeight, int areaHeight, int areaWidth);
	void buildKDTree(Cluster& points, float epsilon, unordered_map<int, vector<int>>& cachedNeighbours);

};

#pragma once
