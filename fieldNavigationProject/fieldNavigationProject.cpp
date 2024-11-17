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
	ui.resolutionComboBox->addItem("4096*3072");


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

	auto start1 = std::chrono::high_resolution_clock::now();
	myImgPro.NormalizedExG(resizedImage, ExGImage);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = end1 - start1;

	/*
		Median filtering is more effective than Gaussian filtering in dealing with salt-and-pepper noise
	*/

	/*auto start11 = std::chrono::high_resolution_clock::now();
	int MedianBlur_kernel_size = 3;
	Mat MedianBlurImg;
	MedianBlurImg = myImgPro.MedianBlur(ExGImage, MedianBlur_kernel_size);
	auto end11 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed11 = end11 - start11;*/

	auto start12 = std::chrono::high_resolution_clock::now();
	auto result_OTSU = myImgPro.OTSU(ExGImage);
	Mat temp = result_OTSU.first;
	Mat OtsuImg = temp.clone();
	NonZeroPixelRatio = result_OTSU.second;
	auto end12 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed12 = end12 - start12;

	/*
		Morphological operations are helpful for eliminating weeds and side branches, but also reduce crop details
	*/
	auto start13 = std::chrono::high_resolution_clock::now();
	Mat MorphImg;
	int flag2 = 0;
	auto result_open = myImgPro.NZPR_to_Erosion_Dilation(NonZeroPixelRatio, resizedImage);
	if (result_open.first > 0 || result_open.second > 0) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, result_open.first, result_open.second);
		flag2 = 1;
	}
	auto end13 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed13 = end13 - start13;


	/*
		The eight-connected algorithm can be employed to further eliminate noise and minor connected components
	*/
	auto start14 = std::chrono::high_resolution_clock::now();
	pair<Mat, vector<int>> result_EC = myImgPro.EightConnectivity(flag2 == 1 ? MorphImg : OtsuImg, 0.7);
	Mat ConnectImg = result_EC.first;
	auto end14 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed14 = end14 - start14;


	//Calculate the x-coordinate of the center row baseline within the crop based on the histogram analysis
	auto start15 = std::chrono::high_resolution_clock::now();
	int centerX = myImgPro.verticalProjectionForCenterX(result_EC.second);//baseline
	//Mat firstHistorImg = result_VPFCX.first;
	auto end15 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed15 = end15 - start15;


	/*
		Using windows to extract features points, reducing data size
	*/
	auto start16 = std::chrono::high_resolution_clock::now();
	CImgPro::Cluster reduce_points;
	Mat featureImg(ConnectImg.size(), CV_8UC1, Scalar(0));
	myImgPro.processImageWithWindow(ConnectImg, featureImg, reduce_points, 8, 8, 1);
	auto end16 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed16 = end16 - start16;


	/*
		Density clustering-based
	*/
	float epsilon = 200;
	int minPts = 4;
	//map<pair<int, int>, pair<float, int>> params{
	//		{{480, 640}, {150, 4}},
	//		{{1080, 1920}, {150, 4}},
	//		{{3072, 4096}, {110, 50}}
	//};
	//auto it = params.find({ CImgPro::imgCols, CImgPro::imgRows });
	//if (it != params.end()) {
	//	epsilon = it->second.first;
	//	minPts = it->second.second;
	//}
	auto start161 = std::chrono::high_resolution_clock::now();
	vector<CImgPro::Cluster> first_cluster_points = myImgPro.KDTreeAcceleratedDBSCAN(reduce_points, epsilon, minPts);
	auto end161 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed161 = end161 - start161;


	auto start17 = std::chrono::high_resolution_clock::now();
	float cof = 0.65;//1
	vector<CImgPro::Cluster> second_cluster_points;
	do
	{
		second_cluster_points = myImgPro.secondClusterBaseOnCenterX(first_cluster_points, centerX, cof);
		cof += 0.05;
	} while (second_cluster_points.size() == 0);
	auto end17 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed17 = end17 - start17;

	//Mat F_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, first_cluster_points);
	//myImgPro.SaveImg(F_ClusterImg);
	//Mat S_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	auto start18 = std::chrono::high_resolution_clock::now();
	//Thresholding segmentation of images
	//Mat HistogramImg;
	double tsd = myImgPro.thresholdingSigmoid(NonZeroPixelRatio, -8.67, 0.354);//0.1-0.9  0.4-0.4
	//double tsd = myImgPro.thresholdingSigmoid(CImgPro::NonZeroPixelRatio, -4.977, 0.3185);//0.04-0.8  0.4-0.4
	myImgPro.verticalProjection(resizedImage, second_cluster_points, tsd);
	myImgPro.retainMainStem(second_cluster_points);
	auto end18 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed18 = end18 - start18;

	Mat MainStemImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);



	auto start19 = std::chrono::high_resolution_clock::now();
	/*
		Second extraction
	*/
	CImgPro::Cluster final_points;
	Mat ExtractImg(MainStemImg.size(), CV_8UC3, Scalar(255, 255, 255));
	myImgPro.processImageWithWindow(MainStemImg, ExtractImg, final_points, 16, 32, 2);
	auto end19 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed19 = end19 - start19;


	auto start20 = std::chrono::high_resolution_clock::now();
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
	auto end20 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed20 = end20 - start20;

	// cv::Mat convert to QImage
	QImage qimg(RansacImg.data, RansacImg.cols, RansacImg.rows, RansacImg.step, QImage::Format_RGB888);

	auto end = std::chrono::high_resolution_clock::now();
	processingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	/*string timefilename = string(std::getenv("USERPROFILE")) + "\\Desktop\\processing_times.txt";
	int processingTime = processingDuration.count();
	myImgPro.saveProcessingTimes(processingTime, timefilename);*/

	return qimg.rgbSwapped();
}


