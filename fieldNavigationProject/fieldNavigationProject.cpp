#include "fieldNavigationProject.h"
#include "ImgPro.h"
#include <QFileDialog>
#include <QDebug>

fieldNavigationProject::fieldNavigationProject(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setWindowTitle(" ");

	ui.startButton->setStyleSheet(
		"QPushButton { "
		"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #e6e6e6, stop:1 #2196F3);"
		"border: 1px solid #a2a2a2; "
		"color: black; "
		"border-radius: 8px; "
		"padding: 5px; "
		"margin-top: 2px; "
		"} "
		"QPushButton:hover { "
		"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #f0f0f0, stop:1 #d8d8d8);"
		"} "
		"QPushButton:pressed { "
		"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #b8b8b8, stop:1 #a2a2a2);"
		"margin-top: 1px; "
		"border: 1px solid #8c8c8c; "
		"}"
	);

	// add resolution
	ui.resolutionComboBox->addItem("640*480");
	ui.resolutionComboBox->addItem("1920*1080");



	connect(ui.resolutionComboBox, SIGNAL(currentIndexChanged(int)),
		this, SLOT(onResolutionChanged(int)));

	connect(ui.startButton, &QPushButton::clicked, this, &fieldNavigationProject::on_startButton_clicked);
}

fieldNavigationProject::~fieldNavigationProject()
{}

cv::Size fieldNavigationProject::onResolutionChanged(int index) {

	QString resolution = ui.resolutionComboBox->itemText(index); // 获取当前选中项的文本

	// 将QString分割成两个字符串，一个是宽度，一个是高度
	QStringList dimensions = resolution.split("*");
	if (dimensions.size() == 2) {
		// 将字符串转换为整数
		int width = dimensions.at(0).toInt();
		int height = dimensions.at(1).toInt();

		// 创建一个cv::Size对象
		cv::Size size(width, height);

		//if (resolution == "640*480") {
		//	qDebug() << 1;
		//}
		//else if (resolution == "1920*1080") {
		//	qDebug() << 2;
		//}
		return size;
	}
}

std::string fieldNavigationProject::openFile() {
	// 打开文件对话框
	qfilename = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
	if (!qfilename.isEmpty()) {
		// 如果用户选择了文件，则处理文件名
		std::string cvFilename = qfilename.toStdString();
		return cvFilename;
	}
	return std::string(); // 如果用户取消了，则返回一个空的字符串
}

void fieldNavigationProject::on_startButton_clicked() {

	std::string filename = openFile();
	if (filename.empty()) {
		isProcessingImage = false;
		return;
	}

	cv::Size targetSize = onResolutionChanged(ui.resolutionComboBox->currentIndex());

	QImage qimg = processImage(filename, targetSize);

	//ui.filePathLabel->setText(qfilename);

	ui.imageLabel->setPixmap(QPixmap::fromImage(qimg));
	ui.imageLabel->setScaledContents(true);

	QString durationText = QString(" %1 ms").arg(processingDuration.count());
	ui.ProcessingTimeLabel->setText(durationText);

	/*QString NZPRText = QString("%1%").arg(NonZeroPixelRatio * 100, 0, 'f', 2);
	ui.NZPRLabel->setText(NZPRText);*/

}

QImage fieldNavigationProject::processImage(cv::String& filename, cv::Size& targetSize) {

	auto start = std::chrono::high_resolution_clock::now();

	Mat inputImage = imread(filename);

	Mat resizedImage;
	cv::resize(inputImage, resizedImage, targetSize, 0, 0, INTER_LINEAR);

	CImgPro::imgCols = resizedImage.cols;
	CImgPro::imgRows = resizedImage.rows;

	CImgPro myImgPro;

	Mat ExGImage(resizedImage.size(), CV_8UC1);
	myImgPro.NormalizedExG(resizedImage, ExGImage);

	/*
		Median filtering is more effective than Gaussian filtering in dealing with salt-and-pepper noise
	*/
	int MedianBlur_kernel_size = 5;
	Mat MedianBlurImg;
	MedianBlurImg = myImgPro.MedianBlur(ExGImage, MedianBlur_kernel_size);


	auto result_OTSU = myImgPro.OTSU(MedianBlurImg);
	Mat temp = result_OTSU.first;
	Mat OtsuImg = temp.clone();
	NonZeroPixelRatio = result_OTSU.second;


	/*
		Morphological operations are helpful for eliminating weeds and side branches, but also reduce crop details
	*/
	Mat MorphImg;
	int flag2 = 0;
	auto result_open = myImgPro.NZPR_to_Erosion_Dilation(NonZeroPixelRatio, resizedImage);
	if (result_open.first > 0 || result_open.second > 0) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, result_open.first, result_open.second);
		flag2 = 1;
	}


	/*
		The eight-connected algorithm can be employed to further eliminate noise and minor connected components
	*/
	pair<Mat, vector<int>> result_EC = myImgPro.EightConnectivity(flag2 == 1 ? MorphImg : OtsuImg, 0.7);
	Mat ConnectImg = result_EC.first;


	//Calculate the x-coordinate of the center row baseline within the crop based on the histogram analysis
	auto result_VPFCX = myImgPro.verticalProjectionForCenterX(result_EC.second);
	Mat firstHistorImg = result_VPFCX.first;
	int centerX = result_VPFCX.second;//baseline


	/*
		Using windows to extract features points, reducing data size
	*/
	CImgPro::Cluster reduce_points;
	Mat featureImg(ConnectImg.size(), CV_8UC1, Scalar(0));
	myImgPro.processImageWithWindow(ConnectImg, featureImg, reduce_points, 8, 8, 1);


	/*
		Density clustering-based
	*/
	float epsilon;
	int minPts;
	if (resizedImage.rows == 480 && resizedImage.cols == 640)
	{
		epsilon = 20;//30
		minPts = 5;//10
	}
	if (resizedImage.rows == 1080 && resizedImage.cols == 1920)
	{
		epsilon = 25;//20
		minPts = 10;// 15
	}
	if (resizedImage.rows == 3072 && resizedImage.cols == 4096)
	{
		epsilon = 110;
		minPts = 50;
	}
	vector<CImgPro::Cluster> first_cluster_points = myImgPro.firstClusterBaseOnDbscan(reduce_points, epsilon, minPts);
	float cof = 0.65;//0.4
	vector<CImgPro::Cluster> second_cluster_points;
	do
	{
		second_cluster_points = myImgPro.secondClusterBaseOnCenterX(first_cluster_points, centerX, cof);
		cof += 0.05;
	} while (second_cluster_points.size() == 0);
	Mat F_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, first_cluster_points);
	Mat S_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	//Thresholding segmentation of images
	Mat HistogramImg;
	double tsd = myImgPro.thresholdingSigmoid(NonZeroPixelRatio, -8.67, 0.354);//0.1-0.9  0.4-0.4
	//double tsd = myImgPro.thresholdingSigmoid(CImgPro::NonZeroPixelRatio, -4.977, 0.3185);//0.04-0.8  0.4-0.4
	HistogramImg = myImgPro.verticalProjection(S_ClusterImg, second_cluster_points, tsd);
	myImgPro.retainMainStem(second_cluster_points);
	Mat MainStemImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	/*
		Second extraction
	*/
	CImgPro::Cluster final_points;
	Mat ExtractImg(MainStemImg.size(), CV_8UC3, Scalar(255, 255, 255));
	myImgPro.processImageWithWindow(MainStemImg, ExtractImg, final_points, 16, 32, 2);


	/*
		fit line
	*/
	Mat RansacImg = resizedImage.clone();
	//Mat RansacImg(ConnectImg.size(), CV_8UC3, Scalar(0, 0, 0));
	if (NonZeroPixelRatio >= 0.1) {
		myImgPro.RANSAC(final_points, 0.155, RansacImg);
	}
	else
	{
		myImgPro.RANSAC(final_points, 0.13, RansacImg);
	}

	// cv::Mat convert to QImage
	QImage qimg(RansacImg.data, RansacImg.cols, RansacImg.rows, RansacImg.step, QImage::Format_RGB888);

	auto end = std::chrono::high_resolution_clock::now();
	processingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	return qimg.rgbSwapped();
}


