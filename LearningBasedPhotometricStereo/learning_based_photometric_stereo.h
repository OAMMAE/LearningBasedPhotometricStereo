#pragma once

#include "stdafx.h"
#include "utility_photometric_stereo.h"

namespace PhotometricStereo
{
	class CLearningBasedPhotometricStereo
	{
	public:
		CLearningBasedPhotometricStereo(
			const int windowSize = 1,
			const int knnCounts = 1,
			const int listReserveSize = 1000000,
			const float knnDistMax = 1000.0) :
			m_windowSize(windowSize),
			m_knnCounts(knnCounts),
			m_listReserveSize(listReserveSize),
			m_knnDistMax(knnDistMax)
		{
		}
		~CLearningBasedPhotometricStereo()
		{
			/////////////////////////////////////////////////////////////
			//////////ポインタで保持しているもののメモリの解放///////////
			/////////////////////////////////////////////////////////////
			if(!m_responseList.empty())
				releaseResponseList(m_responseList);
			if (!m_featureList.empty())
				releaseFeatureList(m_featureList);
			if (!m_ratioFeatureList.empty())
				releaseFeatureList(m_ratioFeatureList);
		}

		void init(std::vector<cv::Vec3d> lightVecList, cv::Vec3d referenceVec, cv::Vec3d observedVec, std::vector<float> sigmaList);

	private:
		int m_windowSize;
		int m_listReserveSize;
		int m_knnCounts;
		float m_knnDistMax;

		FeatureList m_featureList;
		FeatureList m_ratioFeatureList;
		ResponseList m_responseList;
		// featureListをtrainした際のsigma(roughness)の値のindexのリスト [0]が開始index,[1]が範囲内のroughness
		std::vector<std::array<float, 2>> m_sigmaIndexList;
		cv::Mat m_featureListMat;
		cv::flann::Index m_idx;

		std::vector<cv::Vec3d> m_lightVecList;
		cv::Vec3d m_referenceVec;
		cv::Vec3d m_observedVec;
		std::vector<float> m_sigmaList;

	public:
		// Get/Set Property
		std::vector<cv::Vec3d> getLightVecList();
		void setLightVecList(std::vector<cv::Vec3d> lightVecList);
		cv::Vec3d getReferenceVec();
		void setReferenceVec(cv::Vec3d referenceVec);
		cv::Vec3d getObservedVec();
		void setObservedVec(cv::Vec3d observedVec);
		std::vector<float> getSigmaList();
		void setSigmaList(std::vector<float> sigmaList);

		void releaseFeatureList(FeatureList &featureList);
		void releaseResponseList(ResponseList &responseList);
		cv::Mat featureList2Mat(FeatureList &featureList);

		bool syntheticImageLoader4Train(std::string filePath);
		bool syntheticImageLoader4Test(FeatureList & featureList, FeatureList & ratioFeatureList, std::string inFilePath, std::string outFolderPath, double sigma, std::string textureFilePath = "");
		bool realImageLoader4Test(FeatureList & featureList, FeatureList & ratioFeatureList, std::string inFolderPath, std::vector<std::string> fileNameList, std::string referenceFileName, std::string outFolderPath, std::vector<double> lightIntList, double referenceLightInt, bool is16bit = false);
	
		bool train();
		bool test(cv::Mat queryMat, std::string outFolderPath, int testPicRow, int testPicCol, int dataIndex, boost::property_tree::ptree &child);

	private:
		bool measurementImageMaker(cv::Mat normalizedNormalMat, cv::Mat alphaImagedMat, std::string outFilePath, cv::Vec3d observedVec, cv::Vec3d lightVec, float sigma, cv::Mat textureMat = cv::imread(""));
		cv::Mat matNormalizing(cv::Mat inputMat);
		// response/feature Listのindexが与えられたとき、学習に用いたsigma(roughness)の値を返す。
		float getSigma(int index);
	public:
		bool estimatedMapMaker(std::string knnFilePath, std::string outFolderPath, int picRow, int picCol);
	};
}

