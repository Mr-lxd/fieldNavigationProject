#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_fieldNavigationProject.h"
#include <QLabel>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class fieldNavigationProject : public QMainWindow
{
	Q_OBJECT

public:
	fieldNavigationProject(QWidget* parent = nullptr);
	~fieldNavigationProject();

	bool isProcessingImage = false;
	std::chrono::milliseconds processingDuration;
	float NonZeroPixelRatio;
	QString qfilename;

private slots:
	cv::Size onResolutionChanged(int index);
	QImage processImage(cv::String& filename, cv::Size& targetSize);
	std::string openFile();
	void on_startButton_clicked();


private:
	Ui::fieldNavigationClass ui;
};
