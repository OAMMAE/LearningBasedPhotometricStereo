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
	bool test(std::string homeDir)
	{
		using namespace boost::property_tree;
		ptree ptBase;

		int windowSize = 1;
		int knnCounts = 10;
		CLearningBasedPhotometricStereo learningBasedPS(windowSize, knnCounts);
		std::string trainDir = homeDir + "/TrainData/";
		std::string testDir = homeDir + "/TestData/";
		std::string optionDir = homeDir + "/option/";

		std::vector<std::string> roughnessVec;
		//roughnessVec.push_back("0.3");
		roughnessVec.push_back("10.0");
		//roughnessVec.push_back("0.7");
		std::vector<std::string> lightnumVec;
		lightnumVec.push_back("3");
		//lightnumVec.push_back("4");
		//lightnumVec.push_back("5");
		//lightnumVec.push_back("6");
		//lightnumVec.push_back("7");
		//lightnumVec.push_back("8");
		//lightnumVec.push_back("9");
		//lightnumVec.push_back("10");

		std::vector<std::string> trainDataIndexVec;
		//trainDataIndexVec.push_back("0000");
		trainDataIndexVec.push_back("00");
		trainDataIndexVec.push_back("01");
		trainDataIndexVec.push_back("02");
		//trainDataIndexVec.push_back("03");
		//trainDataIndexVec.push_back("04");
		//trainDataIndexVec.push_back("05");
		//trainDataIndexVec.push_back("06");
		//trainDataIndexVec.push_back("07");
		//trainDataIndexVec.push_back("08");
		//trainDataIndexVec.push_back("09");

		std::vector<std::string> testDataIndexVec;
		//testDataIndexVec.push_back("20");
		//testDataIndexVec.push_back("23");
		//testDataIndexVec.push_back("1000");
		testDataIndexVec.push_back("10");
		testDataIndexVec.push_back("11");
		testDataIndexVec.push_back("12");
		testDataIndexVec.push_back("13");
		//testDataIndexVec.push_back("14");
		//testDataIndexVec.push_back("15");
		//testDataIndexVec.push_back("16");
		//testDataIndexVec.push_back("17");
		//testDataIndexVec.push_back("18");
		//testDataIndexVec.push_back("19");

		std::vector<std::string> albedoTestList;
		albedoTestList.push_back("20");

		std::vector<float> sigmaList;
		//sigmaList.push_back(0.0);
		//sigmaList.push_back(0.1);
		//sigmaList.push_back(0.2);
		//sigmaList.push_back(0.3);
		//sigmaList.push_back(0.4);
		//sigmaList.push_back(0.5);
		//sigmaList.push_back(0.6);
		//sigmaList.push_back(0.7);
		sigmaList.push_back(0.0);
		sigmaList.push_back(10.0);
		sigmaList.push_back(20.0);
		sigmaList.push_back(30.0);

		std::vector<cv::Vec3d> lightVecListTrain;
		cv::Vec3d referenceVec;
		cv::Vec3d observedVec(1, 0, 0);

		ptree ptTrain;

		ptree ptTrainName;
		for (int i = 0; i < trainDataIndexVec.size(); i++)
		{
			boost::property_tree::ptree info;
			info.put("", trainDataIndexVec.at(i));
			ptTrainName.push_back(std::make_pair("", info));
		}
		ptTrain.add_child("name", ptTrainName);

		ptree ptTrainRoughness;
		ptTrainRoughness.put("num", sigmaList.size());
		{
			ptree ptTrainRoughnessData;
			for (int i = 0; i < sigmaList.size(); i++)
			{
				boost::property_tree::ptree info;
				info.put("", sigmaList.at(i));
				ptTrainRoughnessData.push_back(std::make_pair("", info));
			}
			ptTrainRoughness.add_child("data", ptTrainRoughnessData);
		}
		ptTrain.add_child("roughness", ptTrainRoughness);

		ptree ptTrainLight;
		{
			////読み込む画像をstringで保持
			std::vector<std::string> readPngList;
			picListLoader(readPngList, optionDir + "filelist96.txt");

			std::vector<std::string> tmpReadPngList;
			tmpReadPngList.assign(&readPngList.at(0), &readPngList.at(atoi(lightnumVec.at(0).c_str())));
			//読み込む画像の番号をintで保持
			std::vector<int> readDataIndexList;

			for (int i = 0; i < readPngList.size(); i++)
			{
				std::string temp = readPngList.at(i).substr(0, 3);
				std::cout << temp << std::endl;
				readDataIndexList.push_back(std::stoi(temp));
			}
			lightVecListLoader(lightVecListTrain, optionDir + "light_directions.txt", readDataIndexList);
		}
		ptTrainLight.put("num", lightVecListTrain.size());
		{
			ptree ptTrainLightData;
			for (int i = 0; i < lightVecListTrain.size(); i++)
			{
				boost::property_tree::ptree info;
				info.put("index", i);
				info.put("x", lightVecListTrain.at(i)[2]);
				info.put("y", lightVecListTrain.at(i)[1]);
				info.put("z", lightVecListTrain.at(i)[0]);
				ptTrainLightData.push_back(std::make_pair("", info));
			}
			ptTrainLight.add_child("data", ptTrainLightData);
		}
		ptTrain.add_child("light", ptTrainLight);

		write_json(homeDir + "train_info.json", ptTrain);


		//std::string inFilePath = "D:/Data/PhotometricStereo/TrainDataBall/snapshot1000.png";

		//読み込む画像をstringで保持
		//std::vector<std::string> readPngList;
		//picListLoader(readPngList, optionDir + "/filelist96.txt");

		for (int lightCount = 0; lightCount < lightnumVec.size(); lightCount++)
		{
			ptree ptLight;
			std::vector<cv::Vec3d> lightVecListTest;

			////読み込む画像をstringで保持
			std::vector<std::string> readPngList;
			picListLoader(readPngList, optionDir + "filelist" + lightnumVec.at(lightCount) + ".txt");

			//読み込む画像の番号をintで保持
			std::vector<int> readDataIndexList;

			//読み込む画像をstringで保持
			std::vector<std::string> tmpReadPngList;

			tmpReadPngList = readPngList;

			//if (atoi(lightnumVec.at(lightCount).c_str()) == 96)
			//	tmpReadPngList = readPngList;
			//else
			//	tmpReadPngList.assign(&readPngList.at(0), &readPngList.at(atoi(lightnumVec.at(lightCount).c_str())));

			for (int i = 0; i < tmpReadPngList.size(); i++)
			{
				std::string temp = tmpReadPngList.at(i).substr(0, 3);
				std::cout << temp << std::endl;
				readDataIndexList.push_back(std::stoi(temp));
			}
			lightVecListLoader(lightVecListTest, optionDir + "light_directions0.txt", readDataIndexList);
			referenceVec = lightVecListTrain.at(0);
			//lightVecListTest = lightVecListTrain;

			learningBasedPS.init(lightVecListTrain, lightVecListTest, referenceVec, observedVec, sigmaList);

			learningBasedPS.searchLightVec();

			boost::timer timer;
			for (int i = 0; i < trainDataIndexVec.size(); i++)
				learningBasedPS.syntheticImageLoader4Train(trainDir + "snapshot" + trainDataIndexVec.at(i) + ".png");
			//learningBasedPS.syntheticImageLoader4Train(inFilePath);
			std::cout << "CalcTime:" << timer.elapsed() << "[s]\n";

			learningBasedPS.train();

			for (int roughnessCount = 0; roughnessCount < roughnessVec.size(); roughnessCount++)
			{
				ptree ptRoughness;

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
				UtilityMethod::str_mkdir(file);

				//file という名前のフォルダを作成する
				file = file + "/PM" + std::to_string(windowSize) + "_" + std::to_string(lightVecListTest.size());
				UtilityMethod::str_mkdir(file);


				//file という名前のフォルダを作成する
				file = file + "/" + std::to_string(knnCounts);
				UtilityMethod::str_mkdir(file);

				std::string albedoFilePath = file + "/albedoTest";
				UtilityMethod::str_mkdir(albedoFilePath);
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

					UtilityMethod::str_mkdir(tempFilePath);
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

					cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList);
					//cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList2);

					learningBasedPS.releaseFeatureList(testfeatureList);
					learningBasedPS.releaseFeatureList(testfeatureList2);

					if (i < testDataIndexVec.size())
						learningBasedPS.test(queryMat, tempFilePath, mPicRow, mPicCol, testDataIndexVec.at(i), ptRoughness);
					else
						learningBasedPS.test(queryMat, tempFilePath, mPicRow, mPicCol, albedoTestList.at(i - testDataIndexVec.size()), ptRoughness);
				}
				ptLight.put("roughness", sSigma);
				ptLight.add_child("result", ptRoughness);
			}
			ptree ptTestLight = learningBasedPS.getLightProp();
			ptLight.add_child("light", ptTestLight);

			ptBase.add_child("test_info", ptLight);
		}

		write_json(testDir + "AvgError.json", ptBase);
		
		return true;
	}

	bool testReal()
	{
		using namespace boost::property_tree;
		ptree ptBase;

		std::string inFilePath = "D:/Data/PhotometricStereo/TrainDataBall/snapshot00.png";
		std::string dataName = "pot1PNG";
		//std::string dataName = "pot2PNG";
		//std::string dataName = "catPNG";
		//std::string dataName = "harvestPNG";
		std::string testDir = "D:/Data/PhotometricStereo/TestDataReal/" + dataName + "/";
		bool is16bit = true;

		//std::string dataName = "cat";
		//std::string testDir = "D:/Data/PhotometricStereo/PSData/" + dataName + "/";
		//bool is16bit = false;


		std::vector<std::string> lightnumVec;
		//lightnumVec.push_back("10");
		lightnumVec.push_back("96");

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
			ptree ptLight;

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
//#pragma omp parallel for
			for (int i = 0; i < lightVecList.size(); i++)
			{
				referenceVec = lightVecList.at(i);
				referenceInt = lightIntList.at(i);
				referencePicName = readPngList.at(i);

				CLearningBasedPhotometricStereo learningBasedPS(1, 30);
				learningBasedPS.init(lightVecList, referenceVec, observedVec, sigmaList);

				std::cout << "training start.\n";
				boost::timer timer;
				learningBasedPS.syntheticImageLoader4Train(inFilePath);
				std::cout << "training have finished. CalcTime:" << timer.elapsed() << "[s]\n";

				learningBasedPS.train();

				//outFolderPath という名前のフォルダを作成する
				std::string outFolderPath = testDir + "/PM" + std::to_string(windowSize) + "_" + std::to_string(lightVecList.size());
				UtilityMethod::str_mkdir(outFolderPath);

				//outFolderPath という名前のフォルダを作成する
				outFolderPath = outFolderPath + "/1_" + std::to_string(i);
				UtilityMethod::str_mkdir(outFolderPath);
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

				learningBasedPS.test(queryMat, outFolderPath, mPicRow, mPicCol, 0, ptLight);
				heatMapMaker(outFolderPath + "GroundTruth_normal.png", outFolderPath + "NearestNeighbor_normal.png", outFolderPath + "HeatMap.png", windowSize, 20);
							
			}
		}
		return true;
	}
}


int main(int argc, char *argv[])
{
	//if (argc < 3)
	//{
	//	std::cout << "invalid argument.\n1:homeDir, 2:num of threads\n";
	//	system("pause");
	//	return 0;
	//}
	//std::string homeDir = argv[1];
	std::string homeDir = "D:/Data/PhotometricStereo/";
	//int ompThreads = std::stoi(argv[2]);
	int ompThreads = 4;

#ifdef _OPENMP
	std::cout << "The number of processors is " << omp_get_num_procs() << std::endl;
	std::cout << "OpenMP : Enabled (Max # of threads = " << omp_get_max_threads() << ")" << std::endl;
	std::cout << "The number of using threads = " << omp_get_num_threads() << std::endl;
	std::cout << "The number of max threads = " << omp_get_max_threads() << std::endl;
	omp_set_num_threads(ompThreads);
	std::cout << "The number of using threads = " << omp_get_num_threads() << std::endl;
	std::cout << "The number of max threads = " << omp_get_max_threads() << std::endl;
#endif

	PhotometricStereo::test(homeDir);
	//PhotometricStereo::testReal();
	std::cout << "finished.\n";
	system("pause");
	return 0;
}

