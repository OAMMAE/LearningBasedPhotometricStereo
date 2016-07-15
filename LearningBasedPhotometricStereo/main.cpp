// main.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
#pragma once

#include "stdafx.h"
#include "learning_based_photometric_stereo.h"
#include "utility.h"
#include "utility_photometric_stereo.h"

#include <omp.h>

namespace PhotometricStereo
{
	bool test()
	{
		CLearningBasedPhotometricStereo learningBasedPS;
		std::string trainDir = "D:/Data/PhotometricStereo/TrainData/";
		std::string testDir = "D:/Data/PhotometricStereo/TestData/";
		std::string optionDir = "D:/Data/PhotometricStereo/TestData/Option/";

		std::vector<std::string> roughnessVec;
		roughnessVec.push_back("0.1");
		std::vector<std::string> lightnumVec;
		lightnumVec.push_back("10");

		std::vector<std::string> testDataIndexVec;
		//testDataIndexVec.push_back("20");
		//testDataIndexVec.push_back("23");
		//testDataIndexVec.push_back("24");

		std::vector<std::string> albedoTestList;
		albedoTestList.push_back("20");

		std::vector<float> sigmaList;
		//sigmaList.push_back(0.0);
		sigmaList.push_back(0.1);
		//sigmaList.push_back(0.2);
		//sigmaList.push_back(0.3);
		//sigmaList.push_back(0.4);

		int windowSize = 1;
		std::vector<cv::Vec3d> lightVecList;
		cv::Vec3d referenceVec;
		cv::Vec3d observedVec(1, 0, 0);

		std::string inFilePath = "D:/Data/PhotometricStereo/TrainDataBall/snapshot00.png";

		for (int lightCount = 0; lightCount < lightnumVec.size(); lightCount++)
		{
			//読み込む画像をstringで保持
			std::vector<std::string> readPngList;
			//読み込む画像の番号をintで保持
			std::vector<int> readDataIndexList;

			picListLoader(readPngList, optionDir + "filelist" + lightnumVec.at(lightCount) + ".txt");
			for (int i = 0; i < readPngList.size(); i++)
			{
				std::string temp = readPngList.at(i).substr(0, 3);
				std::cout << temp << std::endl;
				readDataIndexList.push_back(std::stoi(temp));
			}
			lightVecListLoader(lightVecList, optionDir + "light_directions.txt", readDataIndexList);
			referenceVec = lightVecList.at(0);

			learningBasedPS.init(lightVecList, referenceVec, observedVec, sigmaList);

			boost::timer timer;
			learningBasedPS.syntheticImageLoader4Train(inFilePath);
			std::cout << "CalcTime:" << timer.elapsed() << "[s]\n";

			learningBasedPS.train();

			for (int roughnessCount = 0; roughnessCount < roughnessVec.size(); roughnessCount++)
			{

				std::string sYN;
				std::string sSigma;
				double dSigma;

				std::string sDebug; //debug用にcinでとめるための入力

				std::string sDirName; //ディレクトリ名

				std::cout << "please input GroundTruth sigma\n";
				sSigma = roughnessVec.at(roughnessCount);
				sDirName = "OrenNayar" + sSigma;
				dSigma = std::stod(sSigma);


				std::cout << "Testing data loading..." << std::endl;

				//////////////////////////////////////////
				//////////////testDirの名前///////////////
				//////////////////////////////////////////

				//file という名前のフォルダを作成する
				std::string file = testDir + sDirName;
				UtilityMethod::mkdir(file);

				//file という名前のフォルダを作成する
				file = file + "/PM" + std::to_string(windowSize) + "_" + std::to_string(lightVecList.size());
				UtilityMethod::mkdir(file);


				//file という名前のフォルダを作成する
				file = file + "/1";
				UtilityMethod::mkdir(file);

				std::string albedoFilePath = file + "/albedoTest";
				UtilityMethod::mkdir(albedoFilePath);
				// Testing data loading
#pragma omp parallel for
				for (int i = 0; i < testDataIndexVec.size() + albedoTestList.size(); i++)
				{
					FeatureList testfeatureList;
					FeatureList testfeatureList2;
					//ResponseList testresponseList;

					//file という名前のフォルダを作成する
					std::string tempFilePath = "";

					std::string testfilePath;

					if (i < testDataIndexVec.size())
					{
						testfilePath = testDir + "snapshot" + testDataIndexVec.at(i) + ".png";
						tempFilePath = file + "/" + testDataIndexVec.at(i);
					}
					else
					{
						testfilePath = testDir + "snapshot" + albedoTestList.at(i - testDataIndexVec.size()) + ".png";
						tempFilePath = file + "/albedoTest/" + albedoTestList.at(i - testDataIndexVec.size());
					}

					UtilityMethod::mkdir(tempFilePath);
					tempFilePath = tempFilePath + "/";

					cv::Mat tempMat = cv::imread(testfilePath);
					int c_min, r_min;
					int mPicRow, mPicCol;
					regionExtractor(r_min, c_min, mPicRow, mPicCol, tempMat);
					tempMat.release();

					if (i < testDataIndexVec.size())
						learningBasedPS.syntheticImageLoader4Test(testfeatureList, testfeatureList2, testfilePath, tempFilePath, std::stod(roughnessVec.at(roughnessCount)));
					else
						learningBasedPS.syntheticImageLoader4Test(testfeatureList, testfeatureList2, testfilePath, tempFilePath, std::stod(roughnessVec.at(roughnessCount)), testDir + "Textured" + albedoTestList.at(i - testDataIndexVec.size()) + ".png");

					cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList2);

					learningBasedPS.releaseFeatureList(testfeatureList);
					learningBasedPS.releaseFeatureList(testfeatureList2);

					learningBasedPS.test(queryMat, tempFilePath, mPicRow, mPicCol);
				}
			}
		}
		return true;
	}

	bool testReal()
	{
		std::string inFilePath = "D:/Data/PhotometricStereo/TrainDataBall/snapshot00.png";
		//std::string dataName = "catPNG";
		//std::string testDir = "D:/Data/PhotometricStereo/TestDataReal/" + dataName + "/";
		//bool is16bit = true;

		std::string dataName = "cat";
		std::string testDir = "D:/Data/PhotometricStereo/PSData/" + dataName + "/";
		bool is16bit = false;


		std::vector<std::string> lightnumVec;
		lightnumVec.push_back("10");

		std::vector<float> sigmaList;
		sigmaList.push_back(0.0);
		sigmaList.push_back(0.1);
		sigmaList.push_back(0.2);
		sigmaList.push_back(0.3);
		//sigmaList.push_back(0.4);

		int windowSize = 1;
		cv::Vec3d observedVec(1, 0, 0);

		for (int lightCount = 0; lightCount < lightnumVec.size(); lightCount++)
		{
			std::string referencePicName;
			std::vector<cv::Vec3d> lightVecList;
			cv::Vec3d referenceVec;
			std::vector<double> lightIntList;
			double referenceInt;
			//読み込む画像をstringで保持
			std::vector<std::string> readPngList;
			//読み込む画像の番号をintで保持
			std::vector<int> readDataIndexList;

			picListLoader(readPngList, testDir + "/filelist" + lightnumVec.at(lightCount) + ".txt");
			for (int i = 0; i < readPngList.size(); i++)
			{
				std::string temp = readPngList.at(i).substr(0, 3);
				std::cout << temp << std::endl;
				readDataIndexList.push_back(std::stoi(temp));
			}
			lightVecListLoader(lightVecList, testDir + "/light_directions.txt", readDataIndexList);
			lightIntListLoader(lightIntList, testDir + "/light_intensities.txt", readDataIndexList);

			for (int i = 0; i < lightVecList.size(); i++)
			{
				referenceVec = lightVecList.at(i);
				referenceInt = lightIntList.at(i);
				referencePicName = readPngList.at(i);

				CLearningBasedPhotometricStereo learningBasedPS;
				learningBasedPS.init(lightVecList, referenceVec, observedVec, sigmaList);

				std::cout << "training start.\n";
				boost::timer timer;
				learningBasedPS.syntheticImageLoader4Train(inFilePath);
				std::cout << "training have finished. CalcTime:" << timer.elapsed() << "[s]\n";

				learningBasedPS.train();

				//outFolderPath という名前のフォルダを作成する
				std::string outFolderPath = testDir + "/PM" + std::to_string(windowSize) + "_" + std::to_string(lightVecList.size());
				UtilityMethod::mkdir(outFolderPath);

				//outFolderPath という名前のフォルダを作成する
				outFolderPath = outFolderPath + "/1_" + std::to_string(i);
				UtilityMethod::mkdir(outFolderPath);
				outFolderPath = outFolderPath + "/";

				FeatureList testfeatureList;
				FeatureList testfeatureList2;

				learningBasedPS.realImageLoader4Test(testfeatureList, testfeatureList2, testDir, readPngList, referencePicName, outFolderPath, lightIntList, referenceInt, is16bit);

				cv::Mat gtMat = cv::imread(outFolderPath + "GroundTruth_normal.png");
				int mPicRow, mPicCol;
				int c_min, r_min;
				regionExtractor(r_min, c_min, mPicRow, mPicCol, gtMat);
				gtMat.release();
				std::cout << mPicCol << "," << mPicRow << std::endl;

				cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList2);
				learningBasedPS.releaseFeatureList(testfeatureList);
				learningBasedPS.releaseFeatureList(testfeatureList2);

				learningBasedPS.test(queryMat, outFolderPath, mPicRow, mPicCol);
			}
		}
		return true;
	}
}


int main()
{
#ifdef _OPENMP
	std::cout << "The number of processors is " << omp_get_num_procs() << std::endl;
	std::cout << "OpenMP : Enabled (Max # of threads = " << omp_get_max_threads() << ")" << std::endl;
#endif

	PhotometricStereo::test();
	//PhotometricStereo::testReal();
	std::cout << "finished.\n";
	std::string hoge;
	std::cin >> hoge;
    return 0;
}

