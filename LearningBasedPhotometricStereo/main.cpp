// main.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
#pragma once

#include "stdafx.h"
#include "learning_based_photometric_stereo.h"
#include "utility.h"
#include "utility_photometric_stereo.h"

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
		//testDataIndexVec.push_back("22");
		//testDataIndexVec.push_back("23");
		testDataIndexVec.push_back("24");

		std::vector<std::string> albedoTestList;
		//albedoTestList.push_back("20");

		std::vector<double> sigmaList;
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

			learningBasedPS.init(lightVecList, referenceVec, observedVec);

			for (int i = 0; i < sigmaList.size(); i++)
			{
				boost::progress_timer timer;
				learningBasedPS.syntheticImageLoader4Train(inFilePath, sigmaList.at(i));
			}
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
					int mPicRow, mPicCol;

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
					//if (i < 10)
					//	testfilePath = testDir + "snapshot0" + std::to_string(i) + ".png";
					//else if(i < std::stoi(testdata_quantity))
					//	testfilePath = testDir + "snapshot" + std::to_string(i) + ".png";
					//else
					//	testfilePath = testDir + "snapshot" + albedoTestList.at(i - std::stoi(testdata_quantity)) + ".png";

					cv::Mat tetete = cv::imread(testfilePath);
					int c_min, r_min;
					regionExtractor(r_min, c_min, mPicRow, mPicCol, tetete);


					if (i < testDataIndexVec.size())
						learningBasedPS.syntheticImageLoader4Test(testfeatureList, testfeatureList2, testfilePath, tempFilePath, std::stod(roughnessVec.at(roughnessCount)));
					else
					{
					}
					//releaseFeatureList(testfeatureList);
					cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList2);

					learningBasedPS.releaseFeatureList(testfeatureList);
					learningBasedPS.releaseFeatureList(testfeatureList2);

					cv::Mat tempQueryMat = queryMat(cv::Rect(0, 10050, 10, 500));
					//queryMat = queryMat(cv::Rect(0, 10050, 10, 50));

					//for (int tetete = 0; tetete < 10; tetete++)
					//{
					//	//std::cout << m_featureList.at(tetete)->at(0) << "," << m_featureListMat.at<float>(tetete, 0) << std::endl;
					//	std::cout << testfeatureList.at(10100)->at(tetete) << "," << queryMat.at<float>(10100, tetete) << std::endl;
					//}

					//for (int tetete = 0; tetete < 100; tetete++)
					//{
					//	std::cout << testfeatureList.at(10000 + tetete)->at(0) << std::endl;
					//}

					learningBasedPS.test(queryMat, tempFilePath, mPicRow, mPicCol);
				}
			}
			std::cout << "kokoko\n";
		}
		
		return true;
	}

	bool testReal()
	{
		CLearningBasedPhotometricStereo learningBasedPS;
		std::string inFilePath = "D:/Data/PhotometricStereo/TrainDataBall/snapshot00.png";
		std::string dataName = "catPNG";
		std::string testDir = "D:/Data/PhotometricStereo/TestDataReal/" + dataName + "/";
		//std::string optionDir = "D:/Data/PhotometricStereo/TestData/Option/";

		std::vector<std::string> roughnessVec;
		roughnessVec.push_back("0.1");
		std::vector<std::string> lightnumVec;
		lightnumVec.push_back("10");

		std::vector<std::string> testDataIndexVec;
		//testDataIndexVec.push_back("22");
		//testDataIndexVec.push_back("23");
		testDataIndexVec.push_back("24");

		std::vector<std::string> albedoTestList;
		//albedoTestList.push_back("20");

		std::vector<double> sigmaList;
		//sigmaList.push_back(0.0);
		sigmaList.push_back(0.1);
		//sigmaList.push_back(0.2);
		//sigmaList.push_back(0.3);
		//sigmaList.push_back(0.4);

		int windowSize = 1;
		std::vector<cv::Vec3d> lightVecList;
		cv::Vec3d referenceVec;
		cv::Vec3d observedVec(1, 0, 0);

		for (int lightCount = 0; lightCount < lightnumVec.size(); lightCount++)
		{
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
			referenceVec = lightVecList.at(0);
			lightIntListLoader(lightIntList, testDir + "/light_intensities.txt", readDataIndexList);
			referenceInt = lightIntList.at(0);
			std::string referencePicName = readPngList.at(0);

			learningBasedPS.init(lightVecList, referenceVec, observedVec);

			for (int i = 0; i < sigmaList.size(); i++)
			{
				boost::progress_timer timer;
				learningBasedPS.syntheticImageLoader4Train(inFilePath, sigmaList.at(i));
			}
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

				file = file + "/";

				// Testing data loading

				FeatureList testfeatureList;
				FeatureList testfeatureList2;
				//ResponseList testresponseList;
				int mPicRow, mPicCol;

				std::string testfilePath = testDir;
				
				learningBasedPS.realImageLoader4Test(testfeatureList, testfeatureList2, testfilePath, readPngList, referencePicName, file, lightIntList, referenceInt, true);

				cv::Mat tetete = cv::imread(file + "GroundTruth_normal.png");
				int c_min, r_min;
				regionExtractor(r_min, c_min, mPicRow, mPicCol, tetete);

				std::cout << mPicCol << "," << mPicRow << std::endl;
				cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList2);

				//for (int tetete = queryMat.rows - 500; tetete < queryMat.rows; tetete++)
				//{
				//	std::cout << tetete << ":";
				//	for (int unko = 0; unko < 10; unko++)
				//	{
				//		std::cout << "(" << queryMat.at<float>(tetete, unko) << "," << testfeatureList2.at(tetete)->at(unko) << ")";
				//	}
				//	std::cout << std::endl;
				//}
				//std::cout << testfeatureList2.size() << std::endl;

				learningBasedPS.releaseFeatureList(testfeatureList);
				learningBasedPS.releaseFeatureList(testfeatureList2);

				cv::Mat tempQueryMat = queryMat(cv::Rect(0, 0, 10, 10000));
				//queryMat = queryMat(cv::Rect(0, 10050, 10, 50));

				//for (int tetete = 0; tetete < 10; tetete++)
				//{
				//	//std::cout << m_featureList.at(tetete)->at(0) << "," << m_featureListMat.at<float>(tetete, 0) << std::endl;
				//	std::cout << testfeatureList.at(10100)->at(tetete) << "," << queryMat.at<float>(10100, tetete) << std::endl;
				//}

				//for (int tetete = 0; tetete < 100; tetete++)
				//{
				//	std::cout << testfeatureList.at(10000 + tetete)->at(0) << std::endl;
				//}

				learningBasedPS.test(queryMat, file, mPicRow, mPicCol);
			}
			std::cout << "kokoko\n";
		}

		return true;

	}


}

int main()
{
	//double abc = NULL;
	//double def = 0;
	//if (abc == def)
	//	std::cout << "NULL != 0 true\n";
	//else
	//	std::cout << "oh\n";
	//std::string aa;
	//std::cin >> aa;
	PhotometricStereo::testReal();
	std::string sis;
	std::cin >> sis;
    return 0;
}

