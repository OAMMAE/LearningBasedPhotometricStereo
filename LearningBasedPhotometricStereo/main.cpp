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
		sigmaList.push_back(0.0);
		sigmaList.push_back(0.1);
		//sigmaList.push_back(0.2);
		//sigmaList.push_back(0.3);
		//sigmaList.push_back(0.4);

		int windowSize = 1;

		std::string inFilePath = "D:/Data/PhotometricStereo/TrainDataBall/snapshot00.png";


		for (int lightCount = 0; lightCount < lightnumVec.size(); lightCount++)
		{
			cv::Vec3d mimimisu = (1.0, 0.0, 0.0);
			mimimisu[0] = 1.0;
			learningBasedPS.setObservedVes(mimimisu);
#pragma region Light,ObserveVecの定義

			//読み込む画像をstringで保持
			std::vector<std::string> readPngList;

			//読み込む画像の番号をintで保持
			std::vector<int> readDataIndexList;

			std::ifstream readingFile;
			readingFile.open(optionDir + "filelist" + lightnumVec.at(lightCount) + ".txt", std::ios::in);

			if (readingFile.fail())
			{
				std::cout << "filelist.txtの読み込みに失敗\n";
				return false;
			}

			std::string readLineBuffer;

			while (!readingFile.eof())
			{
				std::getline(readingFile, readLineBuffer);
				if (readLineBuffer.size() == 7)
					readPngList.push_back(readLineBuffer);
			}

			readingFile.close();

			for (int i = 0; i < readPngList.size(); i++)
			{
				std::string temp = readPngList.at(i).substr(0, 3);
				std::cout << temp << std::endl;
				readDataIndexList.push_back(std::stoi(temp));
			}


#pragma region Light_Directions の読み込み

			readingFile.open(optionDir + "light_directions.txt", std::ios::in);

			if (readingFile.fail())
			{
				std::cout << "light_directions.txtの読み込みに失敗\n";
				return false;
			}


			int lineCount = 1;
			int readDataIndex = 0;

			while (!readingFile.eof())
			{
				std::getline(readingFile, readLineBuffer);
				if (readDataIndexList.at(readDataIndex) == lineCount)
				{
					const char delimiter = ' ';
					std::string separatedStringBuffer;
					std::istringstream lineSeparater(readLineBuffer);
					double tempX, tempY, tempZ;
					int i = 0;
					while (getline(lineSeparater, separatedStringBuffer, delimiter))
					{
						if (i == 0)
							tempX = std::stod(separatedStringBuffer);
						else if (i == 1)
							tempY = std::stod(separatedStringBuffer);
						else
							tempZ = std::stod(separatedStringBuffer);
						i++;
					}
					//std::cout << "X,Y,Z = " << tempX << "," << tempY << "," << tempZ << std::endl;
					cv::Vec3d tempL(tempZ, tempY, tempX);
					tempL = tempL / cv::norm(tempL);
					learningBasedPS.m_lightVecList.push_back(tempL);

					readDataIndex++;
					if (readDataIndex == readDataIndexList.size())
						break;
				}
				lineCount++;
			}

			readingFile.close();

#pragma endregion

#pragma endregion

			for (int i = 0; i < sigmaList.size(); i++)
			{
				boost::progress_timer timer;
				FeatureList *tempFeatureList = new FeatureList;
				FeatureList *tempFeatureList2 = new FeatureList;
				ResponseList tempResponseList;
				tempFeatureList->reserve(10000);
				learningBasedPS.readSyntheticImage4Train(*tempFeatureList, *tempFeatureList2, tempResponseList, inFilePath, sigmaList.at(i));
				releaseFeatureList(*tempFeatureList);
				delete tempFeatureList;
				learningBasedPS.m_featureListList.push_back(tempFeatureList2);
				if (i == 0)
					learningBasedPS.m_responseList = tempResponseList;
				else
					releaseResponseList(tempResponseList);
			}
			std::cout << learningBasedPS.m_featureListList.size();
			std::cout << "koko\n";
			learningBasedPS.m_idxList.reserve(1000000);
			learningBasedPS.train();
			learningBasedPS.m_idxList.shrink_to_fit();

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
				file = file + "/PM" + std::to_string(windowSize) + "_" + std::to_string(learningBasedPS.m_lightVecList.size());
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
						learningBasedPS.readSyntheticImage4Test(testfeatureList, testfeatureList2, testfilePath, tempFilePath, std::stod(roughnessVec.at(roughnessCount)));
					else
					{
					}
					releaseFeatureList(testfeatureList);
					cv::Mat queryMat = learningBasedPS.featureList2Mat(testfeatureList2);

					learningBasedPS.test(queryMat, tempFilePath, mPicRow, mPicCol);
				}
			}
			std::cout << "kokoko\n";
		}
		
		return true;
	}
}

int main()
{
	PhotometricStereo::test();
	std::string sis;
	std::cin >> sis;
    return 0;
}

