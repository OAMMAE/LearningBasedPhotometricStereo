#pragma once

#include "stdafx.h"
#include "learning_based_photometric_stereo.h"
#include "utility.h"
#include "utility_photometric_stereo.h"


namespace PhotometricStereo
{
	void CLearningBasedPhotometricStereo::init(std::vector<cv::Vec3d> lightVecList, cv::Vec3d referenceVec, cv::Vec3d observedVec, std::vector<float> sigmaList)
	{
		m_lightVecList = lightVecList;
		m_referenceVec = referenceVec;
		m_observedVec = observedVec;
		m_sigmaList = sigmaList;
		m_featureList.reserve(m_listReserveSize);
		m_ratioFeatureList.reserve(m_listReserveSize);
		m_responseList.reserve(m_listReserveSize);
	}

#pragma region Get/Set Property
	std::vector<cv::Vec3d> CLearningBasedPhotometricStereo::getLightVecList()
	{
		return m_lightVecList;
	}
	void CLearningBasedPhotometricStereo::setLightVecList(std::vector<cv::Vec3d> lightVecList)
	{
		m_lightVecList = lightVecList;
	}
	cv::Vec3d CLearningBasedPhotometricStereo::getReferenceVec()
	{
		return m_referenceVec;
	}
	void CLearningBasedPhotometricStereo::setReferenceVec(cv::Vec3d referenceVec)
	{
		m_referenceVec = referenceVec;
	}
	cv::Vec3d CLearningBasedPhotometricStereo::getObservedVec()
	{
		return m_observedVec;
	}
	void CLearningBasedPhotometricStereo::setObservedVec(cv::Vec3d observedVec)
	{
		m_observedVec = observedVec;
	}
	std::vector<float> CLearningBasedPhotometricStereo::getSigmaList()
	{
		return m_sigmaList;
	}
	void CLearningBasedPhotometricStereo::setSigmaList(std::vector<float> sigmaList)
	{
		m_sigmaList = sigmaList;
	}
#pragma endregion

	/*
	* FeatureList型をMat型に変形する。
	* Matのsizeは(row,col) = (featureの数,各featureの次元)
	*/
	cv::Mat CLearningBasedPhotometricStereo::featureList2Mat(FeatureList & featureList)
	{
		const int nFeatures = featureList.size();
		const int nFeatureDims = featureList.at(0)->size();

		if (sizeof(VAL_TYPE) == sizeof(float))
		{
			cv::Mat m(nFeatures, nFeatureDims, CV_32FC(1));
			for (int j = 0; j < nFeatures; j++)
			{
				for (int i = 0; i < nFeatureDims; i++)
				{
					m.at<float>(j, i) = featureList.at(j)->at(i);
				}
			}
			return m;
		}
		else if (sizeof(VAL_TYPE) == sizeof(double))
		{
			cv::Mat m(nFeatures, nFeatureDims, CV_64FC(1));
			for (int j = 0; j < nFeatures; j++)
			{
				for (int i = 0; i < nFeatureDims; i++)
				{
					m.at<double>(j, i) = featureList.at(j)->at(i);
				}
			}
			return m;
		}
	}

	/*
	* featureListの中身のポインターを全て解放し、featureListのcapacityも0にする
	*/
	void CLearningBasedPhotometricStereo::releaseFeatureList(FeatureList & featureList)
	{
		//responselistの解放
		if (!featureList.empty())
		{
			const int nFeatures = featureList.size();
			for (int i = 0; i < nFeatures; ++i)
			{
				delete featureList.at(i);
			}
		}
		featureList.clear();
		featureList.shrink_to_fit();
	}

	/*
	* responseListの中身のポインターを全て解放し、responseListのcapacityも0にする
	*/
	void CLearningBasedPhotometricStereo::releaseResponseList(ResponseList & responseList)
	{
		//responselistの解放
		if (!responseList.empty())
		{
			const int nResponses = responseList.size();
			for (int i = 0; i < nResponses; ++i)
			{
				delete responseList.at(i);
			}
		}
		responseList.clear();
		responseList.shrink_to_fit();
	}

	bool CLearningBasedPhotometricStereo::measurementImageMaker(cv::Mat normalizedNormalMat, cv::Mat alphaImagedMat, std::string outFilePath, cv::Vec3d observedVec, cv::Vec3d lightVec, float sigma, cv::Mat textureMat)
	{
		if (normalizedNormalMat.data == NULL)
		{
			std::cout << "measurementImageMaker:NormalMatが不正\n";
			return false;
		}
		if (alphaImagedMat.data == NULL)
		{
			std::cout << "measurementImageMaker:alphaImagedMatが不正\n";
			return false;
		}

		bool albedoHandler;
		double rho = 1.0;
		if (textureMat.data == NULL)
			albedoHandler = false;
		else
			albedoHandler = true;

		cv::Mat MatOrenNayar = cv::Mat::ones(normalizedNormalMat.size(), CV_8UC1) * 255;

		for (int y = 0; y < MatOrenNayar.rows; y++)
		{
			for (int x = 0; x < MatOrenNayar.cols; x++)
			{
				if (alphaImagedMat.at<cv::Vec4b>(y, x)[3] != 0)
				{
					cv::Vec3d tempNormalVec = normalizedNormalMat.at<cv::Vec3d>(y, x);
					if (albedoHandler)
					{
						cv::Vec3d tempAlbedoVec = textureMat.at<cv::Vec3b>(y, x);
						rho = rgb2gray(tempAlbedoVec(2), tempAlbedoVec(1), tempAlbedoVec(0)) / 255;
					}
					else
					{
						rho = 1;
					}

					MatOrenNayar.data[y * MatOrenNayar.step + x * MatOrenNayar.elemSize()] = (int)(std::max(orenNayarReflectance(tempNormalVec, observedVec, lightVec, sigma, rho), 0.0) * 255);
				}
			}
		}
		cv::imwrite(outFilePath, MatOrenNayar);
		return true;
	}

	cv::Mat CLearningBasedPhotometricStereo::matNormalizing(cv::Mat inputMat)
	{
		cv::Mat normalizedMat = cv::Mat::zeros(inputMat.rows, inputMat.cols, CV_64FC3);

		for (int i = 0; i < inputMat.rows; i++)
		{
			for (int j = 0; j < inputMat.cols; j++)
			{
				if (inputMat.data[i * inputMat.step + j * inputMat.elemSize()] != 0 || inputMat.data[i * inputMat.step + j * inputMat.elemSize() + 1 * inputMat.elemSize1()] != 0 || inputMat.data[i * inputMat.step + j * inputMat.elemSize() + 2 * inputMat.elemSize1()] != 0)
				{
					cv::Vec3d tempVec;
					for (int k = 0; k < 3; k++)
					{
						//0~255 を-127~128に変換
						tempVec(k) = inputMat.data[i * inputMat.step + j * inputMat.elemSize() + k * inputMat.elemSize1()] - 127;
					}
					for (int k = 0; k < 3; k++)
					{
						normalizedMat.at<cv::Vec3d>(i, j)[k] = tempVec(k) / cv::norm(tempVec);
					}
				}
			}
		}
		return normalizedMat;
	}

	bool CLearningBasedPhotometricStereo::normalMapMaker(std::string knnFilePath, std::string outFolderPath, int picRow, int picCol)
	{
		const int MAT_COL = picCol - m_windowSize + 1;

		cv::Mat output_image = cv::Mat::zeros(picRow - m_windowSize + 1, MAT_COL, CV_8UC3);
		std::ofstream ofs(outFolderPath + "_normal.csv");

		std::ifstream csvknnwriter(knnFilePath);
		std::string str;

		int i = 0;  //col
		int j = 0;  //row

		bool bNN = false;
		while (getline(csvknnwriter, str))
		{
			std::string token;
			std::istringstream stream(str);
			int count = 0;
			int zeroCount = 0; //値がゼロの要素の数
			cv::Vec3d tempNormalVec;
			while (getline(stream, token, ','))
			{
				if (token == "Query point")
				{
					break;
				}

				if (count == 0)
				{
					if (token != "0")
					{
						break;
					}
					else
					{
						bNN = true;
					}
				}
				else if (count == 1)
				{
					//nxをいれる
					tempNormalVec(2) = std::stod(token);
					output_image.data[j * output_image.step + i * output_image.elemSize() + (3 - count) * output_image.elemSize1()] = (int)((std::stod(token) + 1) * 255 / 2);
					ofs << std::to_string(std::stod(token)) << ",";
					if (std::stod(token) == 0)
					{
						zeroCount++;
					}
				}
				else if (count == 2)
				{
					//nyをいれる
					tempNormalVec(1) = std::stod(token);
					output_image.data[j * output_image.step + i * output_image.elemSize() + (3 - count) * output_image.elemSize1()] = (int)((std::stod(token) + 1) * 255 / 2);
					ofs << std::to_string(std::stod(token)) << ",";
					if (std::stod(token) == 0)
					{
						zeroCount++;
					}
				}
				else if (count == 3)
				{
					//nzをいれる
					tempNormalVec(0) = std::stod(token);
					output_image.data[j * output_image.step + i * output_image.elemSize() + (3 - count) * output_image.elemSize1()] = (int)((std::stod(token) + 1) * 255 / 2);
					ofs << std::to_string(std::stod(token)) << std::endl;
					if (std::stod(token) == 0)
					{
						zeroCount++;
					}

					//要素がすべて0のものは枠外とする
					if (zeroCount == 3)
					{
						output_image.data[j * output_image.step + i * output_image.elemSize()] = 0;
						output_image.data[j * output_image.step + i * output_image.elemSize() + 1 * output_image.elemSize1()] = 0;
						output_image.data[j * output_image.step + i * output_image.elemSize() + 2 * output_image.elemSize1()] = 0;
					}
				}
				count++;
			}
			if (bNN)
			{
				i++;
				j = j + (i / MAT_COL);
				i = i % MAT_COL;
				bNN = false;
			}
		}

		cv::Mat alpha_image = alphaImageMaker(output_image, 0);

		cv::imwrite(outFolderPath + "_normal.png", alpha_image);

		return true;
	}

	bool CLearningBasedPhotometricStereo::syntheticImageLoader4Train(std::string filePath)
	{
		cv::Mat inputMat = cv::imread(filePath);
		int nLigths = m_lightVecList.size();
		int nSigmas = m_sigmaList.size();

		if (inputMat.data == NULL)
		{
			std::cout << "syntheticImageLoader4Train:imread の失敗\n";
			return false;
		}
		if (nLigths == 0)
		{
			std::cout << "syntheticImageLoader4Train:Lightの未設定\n";
			return false;
		}
		if (nSigmas == 0)
		{
			std::cout << "syntheticImageLoader4Train:Sigmaの未設定\n";
			return false;
		}

		cv::Mat normalizedMat = matNormalizing(inputMat);

		for (int nSigmaCount = 0; nSigmaCount < nSigmas; nSigmaCount++)
		{
			for (int i = m_windowSize / 2; i < normalizedMat.rows - m_windowSize / 2; i++)
			{
				for (int j = m_windowSize / 2; j < normalizedMat.cols - m_windowSize / 2; j++)
				{
					Response *r = new Response();
					Feature *f = new Feature();
					Feature *f2 = new Feature();

					if (inputMat.data[i * inputMat.step + j * inputMat.elemSize()] != 0 || inputMat.data[i * inputMat.step + j * inputMat.elemSize() + 1 * inputMat.elemSize1()] != 0 || inputMat.data[i * inputMat.step + j * inputMat.elemSize() + 2 * inputMat.elemSize1()] != 0)
					{
						for (int k = 2; k >= 0; k--)
						{
							r->push_back(normalizedMat.at<cv::Vec3d>(i, j)[k]);
						}
						for (int k = -m_windowSize / 2; k <= m_windowSize / 2; k++)
						{
							for (int l = -m_windowSize / 2; l <= m_windowSize / 2; l++)
							{
								cv::Vec3d tempNormalVec = normalizedMat.at<cv::Vec3d>(i + k, j + l);
								//各ピクセルについて法線のz方向が負ならばベクトルの向きを反転する
								if (tempNormalVec(0) < 0)
								{
									tempNormalVec = -tempNormalVec;
								}
								for (int m = 0; m < nLigths; m++)
								{
									//値が不定値にならないための処理、データがない部分は計算を行わない(Edge部分)
									if (cv::norm(tempNormalVec) == 0)
									{
										f->push_back(0.0);
									}
									else
									{
										//法線と光ベクトルの内積が負ならば値を0にする
										f->push_back(std::max(orenNayarReflectance(tempNormalVec, m_observedVec, m_lightVecList[m], m_sigmaList.at(nSigmaCount)), 0.0));
									}
								}

								double referenceIntensity;

								//値が不定値にならないための処理、データがない部分は計算を行わない
								if (cv::norm(tempNormalVec) == 0)
								{
									referenceIntensity = 1;
								}
								else
								{
									//referenceIntensityの最小値は1/255とする
									referenceIntensity = std::max(orenNayarReflectance(tempNormalVec, m_observedVec, m_referenceVec, m_sigmaList.at(nSigmaCount)), (1.0 / 255));
								}

								//f2の入力 f(m)/f(m-1) を計算する
								for (int m = 0; m < nLigths; m++)
								{
									f2->push_back(f->at((m + 1) % nLigths) / referenceIntensity);
								}

							}
						}
						m_featureList.push_back(f);
						m_ratioFeatureList.push_back(f2);
						m_responseList.push_back(r);
					}
					else
					{
						delete r;
						delete f;
						delete f2;
					}
				}
			}
		}

		m_featureList.shrink_to_fit();
		m_ratioFeatureList.shrink_to_fit();
		m_responseList.shrink_to_fit();
		return true;
	}

	bool CLearningBasedPhotometricStereo::syntheticImageLoader4Test(FeatureList & featureList, FeatureList & ratioFeatureList, std::string inFilePath, std::string outFolderPath, double sigma, std::string textureFilePath)
	{
		cv::Mat inputMat = cv::imread(inFilePath);
		cv::Mat textureMat;

		int nLigthCount = m_lightVecList.size();
		bool albedoHandler;
		if (inputMat.data == NULL)
		{
			std::cout << "readSyntheticImage4Test:imreadの失敗\n";
			return false;
		}

		if (textureFilePath != "")
			albedoHandler = true;
		else
			albedoHandler = false;

		int c_min, r_min, newPicCol, newPicRow;
		regionExtractor(r_min, c_min, newPicRow, newPicCol, inputMat);
		inputMat = inputMat(cv::Rect(c_min, r_min, newPicCol, newPicRow));
		cv::Mat alpha_image = alphaImageMaker(inputMat, 0);
		cv::imwrite(outFolderPath + "/GroundTruth_normal.png", alpha_image);

		if (albedoHandler)
		{
			textureMat = cv::imread(textureFilePath);
			textureMat = textureMat(cv::Rect(c_min, r_min, newPicCol, newPicRow));
			cv::Mat tempAlphaImage = alphaImageMaker(textureMat, 0);
			cv::imwrite(outFolderPath + "/GroundTruth_texture.png", tempAlphaImage);
		}

		//////////////////////////////////////////////////////////////////////
		////////////////サイズ修正した画像に対して、処理の開始////////////////
		//////////////////////////////////////////////////////////////////////
		cv::Mat normalizedMat = matNormalizing(inputMat);

		double rho;

		if (albedoHandler)
			measurementImageMaker(normalizedMat, alpha_image, outFolderPath + "/GroundTruth_measurement_reference.png", m_observedVec, m_referenceVec, sigma, textureMat);
		else
			measurementImageMaker(normalizedMat, alpha_image, outFolderPath + "/GroundTruth_measurement_reference.png", m_observedVec, m_referenceVec, sigma);
		for (int i = 0; i < nLigthCount; i++)
		{
			if(albedoHandler)
				measurementImageMaker(normalizedMat, alpha_image, outFolderPath + "/GroundTruth_measurement" + std::to_string(i) + ".png", m_observedVec, m_lightVecList.at(i), sigma, textureMat);
			else
				measurementImageMaker(normalizedMat, alpha_image, outFolderPath + "/GroundTruth_measurement" + std::to_string(i) + ".png", m_observedVec, m_lightVecList.at(i), sigma);
		}
		for (int i = m_windowSize / 2; i < inputMat.rows - m_windowSize / 2; i++)
		{
			for (int j = m_windowSize / 2; j < inputMat.cols - m_windowSize / 2; j++)
			{
				Feature *f = new Feature();
				Feature *f2 = new Feature();

				for (int k = -m_windowSize / 2; k <= m_windowSize / 2; k++)
				{
					for (int l = -m_windowSize / 2; l <= m_windowSize / 2; l++)
					{

						cv::Vec3d tempNormalVec = normalizedMat.at<cv::Vec3d>(i + k, j + l);
						if (albedoHandler)
						{
							cv::Vec3d tempAlbedoVec = textureMat.at<cv::Vec3b>(i + k, j + l);
							rho = rgb2gray(tempAlbedoVec(2), tempAlbedoVec(1), tempAlbedoVec(0)) / 255;
						}
						else
						{
							rho = 1;
						}

						//各ピクセルについて法線のz方向が負ならばベクトルの向きを反転する
						if (tempNormalVec(0) < 0)
						{
							tempNormalVec = -tempNormalVec;
						}
						for (int m = 0; m < nLigthCount; m++)
						{
							//値が不定値にならないための処理、データがない部分は計算を行わない
							if (cv::norm(tempNormalVec) == 0)
							{
								f->push_back(0.0);
							}
							else
							{
								//法線と光ベクトルの内積が負ならば値を0にする
								f->push_back(std::max(orenNayarReflectance(tempNormalVec, m_observedVec, m_lightVecList[m], sigma, rho), 0.0));
							}
						}

						double referenceIntensity;

						//値が不定値にならないための処理、データがない部分は計算を行わない
						if (cv::norm(tempNormalVec) == 0)
						{
							referenceIntensity = 10000;
						}
						else
						{
							referenceIntensity = std::max(orenNayarReflectance(tempNormalVec, m_observedVec, m_referenceVec, sigma, rho), (1.0 / 255));
						}

						//f2の入力 f(m)/f(m-1) を入力する
						for (int m = 0; m < nLigthCount; m++)
						{
							f2->push_back(f->at((m + 1) % nLigthCount) / referenceIntensity);
						}

					}
				}
				featureList.push_back(f);
				ratioFeatureList.push_back(f2);
			}
		}
		return true;
	}

	bool CLearningBasedPhotometricStereo::realImageLoader4Test(FeatureList & featureList, FeatureList & ratioFeatureList, std::string inFolderPath, std::vector<std::string> fileNameList, std::string referenceFileName, std::string outFolderPath, std::vector<double> lightIntList, double referenceLightInt, bool is16bit)
	{
		cv::Mat normalMat = cv::imread(inFolderPath + "Normal_gt.png");
		if (normalMat.data == NULL)
		{
			std::cout << "readRealImage4Test:Normal画像のimreadに失敗\n";
			return false;
		}

		cv::Mat maskMat = cv::imread(inFolderPath + "mask.png", cv::IMREAD_GRAYSCALE);
		if (maskMat.data == NULL)
		{
			std::cout << "readRealImage4Test:Mask画像のimreadに失敗\n";
			return false;
		}

		int nLigthCount = lightIntList.size();

		if (nLigthCount != fileNameList.size())
		{
			std::cout << "readRealImage4Test:入力画像数が不正\n";
			return false;
		}

		std::vector<cv::Mat> testMatList;
		for (int i = 0; i < nLigthCount; i++)
		{
			if (is16bit)
				testMatList.push_back(cv::imread(inFolderPath + fileNameList.at(i), cv::IMREAD_ANYDEPTH));
			else
				testMatList.push_back(cv::imread(inFolderPath + fileNameList.at(i), cv::IMREAD_GRAYSCALE));
		}

		cv::Mat referenceMat;
		if (is16bit)
			referenceMat = cv::imread(inFolderPath + referenceFileName, cv::IMREAD_ANYDEPTH);
		else
			referenceMat = cv::imread(inFolderPath + referenceFileName, cv::IMREAD_GRAYSCALE);

		int c_min, r_min, newPicCol, newPicRow;
		regionExtractor(r_min, c_min, newPicRow, newPicCol, normalMat);
		normalMat = normalMat(cv::Rect(c_min, r_min, newPicCol, newPicRow));
		maskMat = maskMat(cv::Rect(c_min, r_min, newPicCol, newPicRow));
		for (int i = 0; i < nLigthCount; i++)
		{
			testMatList.at(i) = testMatList.at(i)(cv::Rect(c_min, r_min, newPicCol, newPicRow));
			cv::imwrite(outFolderPath + fileNameList.at(i), testMatList.at(i));
			if (is16bit)
				testMatList.push_back(cv::imread(outFolderPath + fileNameList.at(i), cv::IMREAD_ANYDEPTH));
			else
				testMatList.push_back(cv::imread(outFolderPath + fileNameList.at(i), cv::IMREAD_GRAYSCALE));
		}

		referenceMat = referenceMat(cv::Rect(c_min, r_min, newPicCol, newPicRow));
		cv::imwrite(outFolderPath + "ref_" + referenceFileName, referenceMat);
		if (is16bit)
			referenceMat = cv::imread(outFolderPath + "ref_" + referenceFileName, cv::IMREAD_ANYDEPTH);
		else
			referenceMat = cv::imread(outFolderPath + "ref_" + referenceFileName, cv::IMREAD_GRAYSCALE);

		cv::Mat alpha_image = alphaImageMaker(normalMat, 0);
		cv::imwrite(outFolderPath + "GroundTruth_normal.png", alpha_image);
		cv::imwrite(outFolderPath + "mask.png", maskMat);
		maskMat = cv::imread(outFolderPath + "mask.png", cv::IMREAD_GRAYSCALE);

		//realimage から輝度値を取り出す
		for (int i = m_windowSize / 2; i < newPicRow - m_windowSize / 2; i++)
		{
			for (int j = m_windowSize / 2; j < newPicCol - m_windowSize / 2; j++)
			{
				Feature *f = new Feature();
				Feature *f2 = new Feature();
				for (int k = -m_windowSize / 2; k <= m_windowSize / 2; k++)
				{
					for (int l = -m_windowSize / 2; l <= m_windowSize / 2; l++)
					{
						double referenceIntensity;

						//値が不定値にならないための処理、データがない部分は計算を行わない
						if (maskMat.data[(i + k) * maskMat.cols + (j + l)] == 0)
						{
							referenceIntensity = 10000;
						}
						else
						{
							if(is16bit)
								referenceIntensity = referenceMat.at<unsigned short>(i + k, j + l) / referenceLightInt;
							else
								referenceIntensity = referenceMat.at<unsigned char>(i + k, j + l) / referenceLightInt;
						}
						for (int m = 0; m < nLigthCount; m++)
						{
							//値が不定値にならないための処理、データがない部分は計算を行わない
							if (maskMat.data[(i + k) * maskMat.cols + (j + l)] == 0)
							{
								f->push_back(0.0);
							}
							else
							{
								if(is16bit)
									f->push_back(testMatList.at(m).at<unsigned short>(i + k, j + l) / lightIntList.at(m));
								else
									f->push_back(testMatList.at(m).at<unsigned char>(i + k, j + l) / lightIntList.at(m));
							}
						}
						//f2の入力 f(m)/f(m-1) を入力する
						for (int m = 0; m < nLigthCount; m++)
						{
							f2->push_back(f->at((m + 1) % nLigthCount) / referenceIntensity);
						}
					}
				}
				featureList.push_back(f);
				ratioFeatureList.push_back(f2);
			}
		}
		return true;
	}

	bool CLearningBasedPhotometricStereo::train()
	{
		m_featureListMat = featureList2Mat(m_ratioFeatureList);
		releaseFeatureList(m_featureList);
		releaseFeatureList(m_ratioFeatureList);
		std::cout << "kd-tree making start.\n";

		boost::progress_timer timer;
		m_idx.build(m_featureListMat, cv::flann::KDTreeIndexParams(16), cvflann::FLANN_DIST_L1);
		//m_idx.build(m_featureListMat, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L1);
		std::cout << "kd-tree making finished.\n";
		std::cout << "num of features:" << m_featureListMat.rows << std::endl;
		std::cout << "train time:";
		
		return true;
	}

	bool CLearningBasedPhotometricStereo::test(cv::Mat queryMat, std::string outFolderPath, int testPicRow, int testPicCol)
	{
		int overDistCounts = 0;
		int acceptedDistCounts = 0;
		int nSigmas = m_sigmaList.size();
		std::string sSigma;
		if (nSigmas == 1)
		{
			sSigma = std::to_string(m_sigmaList.at(0));
			UtilityMethod::deleteSuffix0(sSigma);
			char tempChar = sSigma.back();
			if (tempChar == ',')
				sSigma.push_back('0');
		}

		//データの出力先のパスを入れる。
		std::ofstream knnwriter;
		if (nSigmas == 1)
			knnwriter.open(outFolderPath + "knn" + std::to_string(m_knnCounts) + "_" + sSigma + ".csv");
		else
			knnwriter.open(outFolderPath + "knn" + std::to_string(m_knnCounts) + ".csv");

		cv::Mat_<int> indices; // dataの何行目か
		cv::Mat_<float> dists; // それぞれどれだけの距離だったか

		double dSumDist = 0;  //NNのEuclid距離の和

		std::cout << "KnnSearch start." << std::endl;
		boost::timer timer;

		//同じfeatureなどを出力した場合に備えて予備に10個のsearchをしておく
		m_idx.knnSearch(queryMat, indices, dists, m_knnCounts + 10);

		std::cout << "KnnSearch have finished.";
		std::cout << "CalcTime:" << timer.elapsed() << "[s]\n";

		std::cout << "query:" << queryMat.cols << std::endl;
		cv::Mat query_max;
		cv::reduce(queryMat, query_max, 1, CV_REDUCE_MAX);

		std::cout << "Results of knnSearch writing start." << std::endl;
		timer.restart();

		for (int j = 0; j < indices.rows; j++) {

			if (dists(j, 0) < m_knnDistMax)
			{
				dSumDist += dists(j, 0);
				acceptedDistCounts++;
			}
			else
			{
				overDistCounts++;
				//std::cout << "over distance\n";
			}

			knnwriter << "Query Point";
			for (int k = 0; k < queryMat.cols; k++)
			{
				//std::cout << k << std::endl;
				knnwriter << "," << queryMat.at<float>(j, k);
			}
			knnwriter << std::endl;

			//query.colsの合計が0ならば（最大が0）探索せず0を入れる
			if (query_max.at<float>(j, 0) == 0)
			{
				for (int k = 0; k < m_knnCounts; k++)
				{
					knnwriter << k << ",0,0,0,0" << std::endl;
				}
			}
			else
			{
				double tempNormalX = NULL; //同じデータが2度続かないようにする
				double tempNormalY = NULL;
				int addition = 0; //同じデータが続いた回数
				int k = 0;

				while (k < m_knnCounts)
				{
					if (tempNormalX != m_responseList.at(indices(j, k + addition))->at(0) && tempNormalY != m_responseList.at(indices(j, k + addition))->at(1))
					{
						//////////////////////////////
						//////エラーに対する処理//////
						//////////////////////////////
						if (dists(j, k + addition) == std::numeric_limits<VAL_TYPE>::denorm_min())
						{
							std::cout << "Error:(i, j, k, addition) = (" << j << ", " << k << ", " << addition << ")\n";
							addition++;
							std::cout << "NEXT_DIST:" << dists(j, k + addition) << ", See." << std::endl;
							while (k < m_knnCounts)
							{
								knnwriter << k << ",0,0,0," << m_knnDistMax << std::endl;
								k++;
							}
						}
						else
						{
							tempNormalX = m_responseList.at(indices(j, k + addition))->at(0);
							tempNormalY = m_responseList.at(indices(j, k + addition))->at(1);
							knnwriter << k << "," << m_responseList.at(indices(j, k + addition))->at(0) << "," << m_responseList.at(indices(j, k + addition))->at(1) << "," << m_responseList.at(indices(j, k + addition))->at(2) << "," << dists(j, k + addition) << std::endl;
							k++;
						}
					}
					else
					{
						addition++;
					}
				}
			}
		}

		double dAvgDist = dSumDist / acceptedDistCounts;

		knnwriter.close();
		std::cout << "knn result writing is finished." << "  CalcTime:" << timer.elapsed() << "[s]\n" << std::endl;

		std::string inputknn;
		if(nSigmas == 1)
			inputknn = outFolderPath + "knn" + std::to_string(m_knnCounts) + "_" + sSigma + ".csv";
		else
			inputknn = outFolderPath + "knn" + std::to_string(m_knnCounts) + ".csv";

		std::string output = outFolderPath + "NearestNeighbor";

		std::cout << "Image, Roughness : Png making start!" << std::endl;

		timer.restart();
		normalMapMaker(inputknn, output, testPicRow, testPicCol);
		std::cout << "Image, Roughness : Png making finished!" << "  CalcTime:" << timer.elapsed() << "[s]\n" << std::endl;

		std::string sGroundTruthPath = outFolderPath + "GroundTruth_normal.png";
		std::string sEstimatePath = outFolderPath + "NearestNeighbor_normal.png";

		cv::Mat matEstimateNormalMap = cv::imread(sEstimatePath);
		double dSmoothness = calculateSmoothness(matEstimateNormalMap);
		double dEvaluation = evaluate(sGroundTruthPath, sEstimatePath, m_windowSize);
		std::cout << "精度 : (DataNumber , AvgError(rad) , Smoothness , EuclidDistance) : ( " << dEvaluation << " , " << dSmoothness << "," << dAvgDist << " )" << std::endl;
		std::cout << "over distance:" << std::to_string(overDistCounts) << "\n";

		std::ofstream evaluationWriter;
		if(nSigmas == 1)
			evaluationWriter.open(outFolderPath + "AvgError_" + sSigma + ".csv");
		else
			evaluationWriter.open(outFolderPath + "AvgError.csv");

		evaluationWriter << "AvgError(rad),AvgError(deg),Smoothness,EuclidDistance" << std::endl;
		evaluationWriter <<  dEvaluation << "," << dEvaluation * 180 / M_PI << "," << dSmoothness << "," << dAvgDist << std::endl;

		return true;
	}


}


