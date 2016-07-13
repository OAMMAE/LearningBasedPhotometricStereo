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
			const int knnCounts = 30,
			const float knnDistMax = 1000) :
			m_windowSize(windowSize),
			m_knnCounts(knnCounts),
			m_knnDistMax(knnDistMax)
		{
		}
		CLearningBasedPhotometricStereo::~CLearningBasedPhotometricStereo()
		{
			/////////////////////////////////////////////////////////////
			//////////É|ÉCÉìÉ^Ç≈ï€éùÇµÇƒÇ¢ÇÈÇ‡ÇÃÇÃÉÅÉÇÉäÇÃâï˙///////////
			/////////////////////////////////////////////////////////////
			releaseResponseList(m_responseList);

			if (!m_featureListList.empty())
			{
				const int nFeatureLists = m_featureListList.size();
				for (int i = 0; i < nFeatureLists; ++i)
				{
					releaseFeatureList(*m_featureListList.at(i));
					delete m_featureListList.at(i);
				}
				m_featureListList.clear();
				m_featureListList.shrink_to_fit();
			}
		}

	private:
		int m_windowSize;
		int m_knnCounts;
		float m_knnDistMax;
		cv::Vec3d m_observedVec;

	public:
		std::vector<FeatureList*> m_featureListList;
		std::vector<cv::Mat> m_featureListMatList;
		std::vector<cv::flann::Index> m_idxList;
		ResponseList m_responseList;

		cv::flann::Index m_idxList2[2];

		std::vector<cv::Vec3d> m_lightVecList;
		std::vector<double> m_lightIntList;
		cv::Vec3d m_referenceVec;
		double m_referenceInt;

		void setObservedVes(cv::Vec3d observedVec);
		cv::Mat featureList2Mat(FeatureList &featureList);
		bool readSyntheticImage4Train(FeatureList &featureList, FeatureList &ratioFeatureList, ResponseList &responseList, std::string filePath, double sigma);
		bool readSyntheticImage4Test(FeatureList &featureList, FeatureList &ratioFeatureList, std::string inFilePath, std::string outFolderPath, double sigma, bool albedoHandler = false, std::string textureFilePath = "");
		bool readRealImage4Test(FeatureList &featureList, FeatureList &ratioFeatureList, std::string inFolderPath, std::vector<std::string> filePathList, std::string referenceFilePath, std::string outFilePath, int nLigthCount, std::vector<double> lightIntList, double referenceLightInt, bool is16bit = false);
	
		bool train();
		bool test(cv::Mat queryMat, std::string outFolderPath, int testPicRow, int testPicCol);

	private:
		cv::Mat matNormalizing(cv::Mat inputMat);
		bool normalMapMaker(std::string knnFilePath, std::string outFolderPath, int picRow, int picCol);
	};
}

