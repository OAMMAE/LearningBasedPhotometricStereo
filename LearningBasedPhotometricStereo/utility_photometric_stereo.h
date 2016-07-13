#pragma once

#include "stdafx.h"
#include "common.h"

namespace PhotometricStereo
{
	void matPropertyPrinter(cv::Mat mat);

	bool regionExtractor(int &r_min, int &c_min, int &newPicRow, int &newPicCol, cv::Mat inputMat);
	cv::Mat alphaImageMaker(cv::Mat inputMat, int transValue);

	double orenNayarReflectance(cv::Vec3d normalVec, cv::Vec3d observeVec, cv::Vec3d lightVec, double sigma, double rho = 1, double measurementIllumination = 1);
	double rgb2gray(double r, double g, double b);

	bool picListLoader(std::vector<std::string> &picNameList, std::string inFilePath);
	bool lightVecListLoader(std::vector<cv::Vec3d> &lightVecList, std::string inFilePath, std::vector<int> readDataIndexList);
	bool lightIntListLoader(std::vector<double> &lightIntList, std::string inFilePath, std::vector<int> readDataIndexList);



	double evaluate(std::string groundTruth, std::string estimate, const int WSIZE);
	bool createHeatMap(std::string groundTruth, std::string estimate, std::string outFileName, const int WSIZE, const double MAX_DEG);
	bool maskImageMaker(std::string inputPic, std::string outputPic, const int THRESHOLD);

	double calculateSmoothness(cv::Mat matNormalMap);

	bool saveFeatureList(const FeatureList &featureList, double roughness, std::string outFilePath, int row = 0, int col = 0);
	bool saveResponseList(const ResponseList &responseList, double roughness, std::string outFilePath, int row = 0, int col = 0);
	bool txt2Png(std::string inFilePath, std::string outFilePath);


	//template <class List>	void clearList(List &list);

	template<typename Type>
	bool writeTxt(const std::string filename, const cv::Mat_<Type> & mat);

	bool writeTxt_float(const std::string filename, const cv::Mat & mat);
	template<typename Type>
	bool readTxt(const std::string filename, cv::Mat_<Type> & mat);
}