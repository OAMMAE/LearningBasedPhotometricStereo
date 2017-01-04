#pragma once

#include "stdafx.h"
#include "utility_photometric_stereo.h"
#include "utility.h"

namespace PhotometricStereo
{
	/*
	* �摜����object�����݂���̈����肷��B
	*/
	bool regionExtractor(int &r_min, int &c_min, int &newPicRow, int &newPicCol, cv::Mat inputMat)
	{
		int r_max = inputMat.rows;
		r_min = 0;
		int c_max = inputMat.cols;
		c_min = 0;

		cv::Mat test_col;   //����(�s������{��)
		cv::Mat test_row;   //��s��(�������{��)
		cv::reduce(inputMat, test_col, 1, CV_REDUCE_MAX);
		cv::reduce(inputMat, test_row, 0, CV_REDUCE_MAX);

		int mRow = inputMat.rows;
		int mCol = inputMat.cols;
		bool bMinFin = false; //min�̍X�V�������������ۂ�
		for (int i = 0; i < mRow; i++)
		{
			if (test_col.at<cv::Vec3b>(i, 0)[0] == 0 && !bMinFin)
			{
				if (r_min < i)
					r_min = i;
			}
			else if (!bMinFin)
			{
				bMinFin = true;
			}
			else if (test_col.at<cv::Vec3b>(i, 0)[0] == 0 && bMinFin)
			{
				if (r_max > i)
					r_max = i;
			}
		}
		bMinFin = false;
		for (int i = 0; i < mCol; i++)
		{
			if (test_row.at<cv::Vec3b>(0, i)[0] == 0 && !bMinFin)
			{
				if (c_min < i)
					c_min = i;
			}
			else if (!bMinFin)
			{
				bMinFin = true;
			}
			else if (test_row.at<cv::Vec3b>(0, i)[0] == 0 && bMinFin)
			{
				if (c_max > i)
					c_max = i;
			}
		}

		newPicRow = r_max - r_min;
		newPicCol = c_max - c_min;

		return true;
	}

	/*
	* inputMat�ɂ��āA3channel�S�Ă�transValue�̕����𓧉ߐF�ɐݒ肵��alphaImage��Mat��Ԃ��B
	*/
	cv::Mat alphaImageMaker(cv::Mat inputMat, int transValue)
	{
		cv::Mat alpha_image = cv::Mat(inputMat.size(), CV_8UC3);
		cv::cvtColor(inputMat, alpha_image, CV_RGB2RGBA);
		for (int y = 0; y < alpha_image.rows; ++y) {
			for (int x = 0; x < alpha_image.cols; ++x) {
				cv::Vec4b px = alpha_image.at<cv::Vec4b>(y, x);
				if (px[0] == transValue && px[1] == transValue && px[2] == transValue) {
					px[3] = 0;
					alpha_image.at<cv::Vec4b>(y, x) = px;
				}
			}
		}
		return alpha_image;
	}

	/*
	* Oren-Nayar���f�����g�����ꍇ�̔��ˌ����x(Measurement)���v�Z���A�Ԃ��B
	*/
	double orenNayarReflectance(cv::Vec3d normalVec, cv::Vec3d observeVec, cv::Vec3d lightVec, double sigma, double rho, double measurementIllumination)
	{
		//////////////////////////////////////////////////////////////////////
		//////////////normal�ɑ΂��鐳�ˉe�x�N�g�����v�Z����//////////////////
		//////////////////////////////////////////////////////////////////////
		double temp = normalVec.dot(observeVec) / cv::norm(normalVec);
		cv::Vec3d orthogonalObserveVec = observeVec - temp * normalVec;

		temp = normalVec.dot(lightVec) / cv::norm(normalVec);
		cv::Vec3d orthogonalLightVec = lightVec - temp * normalVec;

		double cos_Phi = orthogonalObserveVec.dot(orthogonalLightVec) / (cv::norm(orthogonalLightVec) * cv::norm(orthogonalObserveVec));

		/////////////////////////////////////////////////////////////////////
		/////////////////////////A,B���Z�o///////////////////////////////////
		/////////////////////////////////////////////////////////////////////
		const double A = 1 - (0.5 * (pow(sigma, 2) / (pow(sigma, 2) + 0.33)));
		const double B = 0.45 * (pow(sigma, 2) / (pow(sigma, 2) + 0.09));

		double cos_ThetaI = normalVec.dot(lightVec) / (cv::norm(normalVec) * cv::norm(lightVec));
		double cos_ThetaR = normalVec.dot(observeVec) / (cv::norm(normalVec) * cv::norm(observeVec));

		double ThetaI = acos(cos_ThetaI);
		double ThetaR = acos(cos_ThetaR);

		double alpha = std::max(ThetaI, ThetaR);
		double beta = std::min(ThetaI, ThetaR);

		double result = rho * cos_ThetaI * (A + (B * std::max(0.0, cos_Phi) * sin(alpha) * tan(beta))) * measurementIllumination;

		return result;
	}

	/*
	* RGB ���� GrayScale �ɕϊ����鎮
	*/
	double rgb2gray(double r, double g, double b)
	{
		return 0.299 * r + 0.587 * g + 0.114 * b;
	}

	/*
	* picNameList�ɓǂݍ��މ摜��filename���i�[ filename��"000.png"�Ƃ���7��
	*/
	bool picListLoader(std::vector<std::string>& picNameList, std::string inFilePath)
	{
		std::ifstream readingFile;
		readingFile.open(inFilePath, std::ios::in);

		if (readingFile.fail())
		{
			std::cout << "picListLoader:filelist.txt�̓ǂݍ��݂Ɏ��s\n";
			return false;
		}

		std::string readLineBuffer;

		while (!readingFile.eof())
		{
			std::getline(readingFile, readLineBuffer);
			if (readLineBuffer.size() == 7)
				picNameList.push_back(readLineBuffer);
		}
		readingFile.close();
		return true;
	}

	bool lightVecListMaker(std::vector<cv::Vec3d>& lightVecList, double maxTheta)
	{
		for (int theta = 0; theta < maxTheta; theta++)
		{
			double thetaRad = theta * M_PI / 180.0;
			for (int phi = 0; phi < 360; phi++)
			{
				double phiRad = phi * M_PI / 180.0;
				double tempX, tempY, tempZ;

				tempX = sin(thetaRad) * cos(phiRad);
				tempY = sin(thetaRad) * sin(phiRad);
				tempZ = cos(thetaRad);

				cv::Vec3d tempL(tempZ, tempY, tempX);
				tempL = tempL / cv::norm(tempL);
				lightVecList.push_back(tempL);
			}
		}
		return true;
	}

	/*
	* lightVecList�Ɍ����x�N�g�����i�[,delimiter ��' '��','
	*/
	bool lightVecListLoader(std::vector<cv::Vec3d>& lightVecList, std::string inFilePath, std::vector<int> readDataIndexList)
	{
		std::ifstream readingFile;
		readingFile.open(inFilePath, std::ios::in);

		if (readingFile.fail())
		{
			std::cout << "lightVecListLoader:light_directions.txt�̓ǂݍ��݂Ɏ��s\n";
			return false;
		}

		int lineCount = 1;
		int readDataIndex = 0;
		std::string readLineBuffer;

		while (!readingFile.eof())
		{
			std::getline(readingFile, readLineBuffer);
			if (readDataIndexList.at(readDataIndex) == lineCount)
			{
				std::vector<std::string> separatedBufferList;
				boost::split(separatedBufferList, readLineBuffer, boost::is_any_of(L", "));

				double tempX, tempY, tempZ;
				tempX = std::stod(separatedBufferList.at(0));
				tempY = std::stod(separatedBufferList.at(1));
				tempZ = std::stod(separatedBufferList.at(2));

				cv::Vec3d tempL(tempZ, tempY, tempX);
				tempL = tempL / cv::norm(tempL);
				lightVecList.push_back(tempL);

				readDataIndex++;
				if (readDataIndex == readDataIndexList.size())
					break;
			}
			lineCount++;
		}
		readingFile.close();
		return true;
	}

	/*
	* lightIntList�Ɍ����̋��x���i�[,delimiter ��' '
	*/
	bool lightIntListLoader(std::vector<double>& lightIntList, std::string inFilePath, std::vector<int> readDataIndexList)
	{
		std::ifstream readingFile;
		readingFile.open(inFilePath, std::ios::in);

		if (readingFile.fail())
		{
			std::cout << "lightVecListLoader:light_directions.txt�̓ǂݍ��݂Ɏ��s\n";
			return false;
		}

		int lineCount = 1;
		int readDataIndex = 0;
		std::string readLineBuffer;

		while (!readingFile.eof())
		{
			std::getline(readingFile, readLineBuffer);
			if (readDataIndexList.at(readDataIndex) == lineCount)
			{
				const char delimiter = ' ';
				std::string separatedStringBuffer;
				std::istringstream lineSeparater(readLineBuffer);
				double tempR, tempG, tempB;
				int i = 0;
				while (getline(lineSeparater, separatedStringBuffer, delimiter))
				{
					if (i == 0)
						tempR = std::stod(separatedStringBuffer);
					else if (i == 1)
						tempG = std::stod(separatedStringBuffer);
					else
						tempB = std::stod(separatedStringBuffer);
					i++;
				}
				double tempIntensity = rgb2gray(tempR, tempG, tempB);
				lightIntList.push_back(tempIntensity);

				readDataIndex++;
				if (readDataIndex == readDataIndexList.size())
					break;
			}
			lineCount++;
		}

		readingFile.close();
		return true;
	}

	/*
	* groundTruth �� estimate �̓�摜��normal�̍����v�Z���A�Ԃ��B(rad)
	*/
	double evaluate(std::string groundTruth, std::string estimate, const int WSIZE)
	{
		cv::Mat matTemp, matEstimate;
		matTemp = cv::imread(groundTruth);
		matEstimate = cv::imread(estimate);
		cv::Mat matGroundTruth(matTemp, cv::Rect(WSIZE / 2, WSIZE / 2, matEstimate.cols, matEstimate.rows));

		//cv::imwrite(groundTruth + ".png", matGroundTruth);

		int nNormalCount = 0;
		double result = 0;

		for (int y = 0; y < matEstimate.rows; y++)
		{
			for (int x = 0; x < matEstimate.cols; x++)
			{
				//�g�O�łȂ����̔���
				if (matGroundTruth.data[y * matGroundTruth.step + x * matGroundTruth.elemSize()] != 0 && matGroundTruth.data[y * matGroundTruth.step + x * matGroundTruth.elemSize() + matGroundTruth.elemSize1()] != 0)
				{
					cv::Vec3d groundTruthVec, estimateVec;

					for (int i = 0; i < 3; i++)
					{
						groundTruthVec(i) = matGroundTruth.data[y * matGroundTruth.step + x * matGroundTruth.elemSize() + i * matGroundTruth.elemSize1()] - 127;
						estimateVec(i) = matEstimate.data[y * matEstimate.step + x * matEstimate.elemSize() + i * matEstimate.elemSize1()] - 127;
					}

					groundTruthVec = groundTruthVec / cv::norm(groundTruthVec);
					estimateVec = estimateVec / cv::norm(estimateVec);

					double dCos = groundTruthVec.dot(estimateVec) / cv::norm(groundTruthVec) / cv::norm(estimateVec);
					if (dCos > 1)
					{
						dCos = 1;
					}
					if (dCos < -1)
					{
						dCos = -1;
					}
					double dError;



					if (dCos == 1)
					{
						dError = 0;
					}
					else
					{
						dError = acos(dCos);
					}

					result += dError;
					nNormalCount++;
				}
			}
		}

		//for debug
		//std::cout << "count ,error:" << nNormalCount << " ," << result << std::endl;

		result = result / nNormalCount;
		return result;
	}

	/*
	* groundTruth �� estimate �̓�摜��normal�̍���HeatMap�ɂ���
	*/
	bool heatMapMaker(std::string groundTruth, std::string estimate, std::string outFileName, const int WSIZE, const double MAX_DEG)
	{
		cv::Mat matTemp, matEstimate;
		matTemp = cv::imread(groundTruth);
		matEstimate = cv::imread(estimate);
		cv::Mat matGroundTruth(matTemp, cv::Rect(WSIZE / 2, WSIZE / 2, matEstimate.cols, matEstimate.rows));

		cv::Mat matHeatMap = cv::Mat::zeros(matEstimate.rows, matEstimate.cols, CV_8UC1);

		int nNormalCount = 0;

		double maxError = 0;

		for (int y = 0; y < matEstimate.rows; y++)
		{
			for (int x = 0; x < matEstimate.cols; x++)
			{
				//�g�O�łȂ����̔���
				if (matGroundTruth.data[y * matGroundTruth.step + x * matGroundTruth.elemSize()] != 0 && matGroundTruth.data[y * matGroundTruth.step + x * matGroundTruth.elemSize() + matGroundTruth.elemSize1()] != 0)
				{
					cv::Vec3d groundTruthVec, estimateVec;

					for (int i = 0; i < 3; i++)
					{
						groundTruthVec(i) = matGroundTruth.data[y * matGroundTruth.step + x * matGroundTruth.elemSize() + i * matGroundTruth.elemSize1()] - 127;
						estimateVec(i) = matEstimate.data[y * matEstimate.step + x * matEstimate.elemSize() + i * matEstimate.elemSize1()] - 127;
					}

					groundTruthVec = groundTruthVec / cv::norm(groundTruthVec);
					estimateVec = estimateVec / cv::norm(estimateVec);

					double dCos = groundTruthVec.dot(estimateVec) / cv::norm(groundTruthVec) / cv::norm(estimateVec);
					if (dCos > 1)
					{
						dCos = 1;
					}
					if (dCos < -1)
					{
						dCos = -1;
					}
					double dError;



					if (dCos == 1)
					{
						dError = 0;
					}
					else
					{
						dError = acos(dCos);
					}


					dError = dError * 180 / M_PI;

					if (dError > maxError)
						maxError = dError;

					dError = dError / MAX_DEG;

					matHeatMap.data[y * matHeatMap.step + x] = (int)(dError * 255);
				}
			}
		}

		cv::imwrite(outFileName, matHeatMap);

		std::cout << estimate << std::endl;
		std::cout << maxError << std::endl << std::endl;

		return true;

	}

	/*
	*
	*/
	bool maskImageMaker(std::string inputPic, std::string outputPic, const int THRESHOLD)
	{
		cv::Mat in, out;
		in = cv::imread(inputPic, 0);

		out = cv::Mat::ones(in.rows, in.cols, CV_8UC1) * 255;

		for (int y = 0; y < in.rows; y++)
		{
			for (int x = 0; x < in.cols; x++)
			{
				if (in.data[y * in.cols + x] < THRESHOLD)
				{
					out.data[y * out.cols + x] = 0;
				}
			}
		}

		cv::imwrite(outputPic, out);
		return true;
	}

	/*
	* matNormalMap��smoothness���v�Z���Ԃ��B
	* smoothness�͗אڍ��Ƃ̊p�x�̍��Ōv�Z����B
	*/
	double calculateSmoothness(cv::Mat matNormalMap)
	{
		const int HEIGHT = matNormalMap.rows;
		const int WIDTH = matNormalMap.cols;
		double result = 0;

		for (int y = 0; y < HEIGHT - 1; y++)
		{
			for (int x = 0; x < WIDTH - 1; x++)
			{
				cv::Vec3d tempNormalVec, tempNormalVecNeighborHorizon, tempNormalVecNeighborVertical;
				for (int i = 0; i < 3; i++)
				{
					tempNormalVec(i) = matNormalMap.data[y * matNormalMap.step + x * matNormalMap.elemSize() + i * matNormalMap.elemSize1()] - 127;
					tempNormalVecNeighborHorizon(i) = matNormalMap.data[y * matNormalMap.step + (x + 1) * matNormalMap.elemSize() + i * matNormalMap.elemSize1()] - 127;
					tempNormalVecNeighborVertical(i) = matNormalMap.data[(y + 1) * matNormalMap.step + x * matNormalMap.elemSize() + i * matNormalMap.elemSize1()] - 127;
				}
				if (tempNormalVec(0) != -127 || tempNormalVec(1) != -127 || tempNormalVec(2) != -127)
				{
					tempNormalVec = tempNormalVec / cv::norm(tempNormalVec);
					if (tempNormalVecNeighborHorizon(0) != -127 || tempNormalVecNeighborHorizon(1) != -127 || tempNormalVecNeighborHorizon(2) != -127)
					{
						tempNormalVecNeighborHorizon = tempNormalVecNeighborHorizon / cv::norm(tempNormalVecNeighborHorizon);
						double dCos = tempNormalVec.dot(tempNormalVecNeighborHorizon) / cv::norm(tempNormalVec) / cv::norm(tempNormalVecNeighborHorizon);
						double dTheta;
						if (dCos >= 1)
						{
							dTheta = 0;
						}
						else
						{
							dTheta = acos(dCos);
						}
						result += dTheta;
					}
					if (tempNormalVecNeighborVertical(0) != -127 || tempNormalVecNeighborVertical(1) != -127 || tempNormalVecNeighborVertical(2) != -127)
					{
						tempNormalVecNeighborVertical = tempNormalVecNeighborVertical / cv::norm(tempNormalVecNeighborVertical);
						double dCos = tempNormalVec.dot(tempNormalVecNeighborVertical) / cv::norm(tempNormalVec) / cv::norm(tempNormalVecNeighborVertical);
						double dTheta;
						if (dCos >= 1)
						{
							dTheta = 0;
						}
						else
						{
							dTheta = acos(dCos);
						}
						result += dTheta;
					}
				}
			}
		}
		return result;
	}

	/*
	* mat�̃v���p�e�B���R���\�[���ɏ����o��
	*/
	void matPropertyPrinter(cv::Mat mat)
	{
		// �������i�摜�Ȃ̂ŏc�E����2�����j
		std::cout << "dims: " << mat.dims << std::endl;
		// �T�C�Y�i2�����̏ꍇ�j
		std::cout << "size[]: " << mat.size().width << "," << mat.size().height << std::endl;
		// �r�b�g�[�xID
		std::cout << "depth (ID): " << mat.depth() << "(=" << CV_8U << ")" << std::endl;
		// �`�����l����
		std::cout << "channels: " << mat.channels() << std::endl;
		// �i�����`�����l�����琬��j1�v�f�̃T�C�Y [�o�C�g�P��]
		std::cout << "elemSize: " << mat.elemSize() << "[byte]" << std::endl;
		// 1�v�f����1�`�����l�����̃T�C�Y [�o�C�g�P��]
		std::cout << "elemSize1 (elemSize/channels): " << mat.elemSize1() << "[byte]" << std::endl;
		std::cout << "type (ID): " << mat.type() << "(=" << CV_8UC3 << ")" << std::endl;
		// �v�f�̑���
		std::cout << "total: " << mat.total() << std::endl;
		// �X�e�b�v�� [�o�C�g�P��]
		// �^�C�v
		std::cout << "step: " << mat.step << "[byte]" << std::endl;
		// 1�X�e�b�v���̃`�����l������
		std::cout << "step1 (step/elemSize1): " << mat.step1() << std::endl;
		// �f�[�^�͘A�����H
		std::cout << "isContinuous: " << (mat.isContinuous() ? "true" : "false") << std::endl;
		// �����s�񂩁H
		std::cout << "isSubmatrix: " << (mat.isSubmatrix() ? "true" : "false") << std::endl;
		// �f�[�^�͋󂩁H
		std::cout << "empty: " << (mat.empty() ? "true" : "false") << std::endl;

	}

	/*
	* featureList�̒��g��outFilePath�ɏ����o��
	*/
	bool saveFeatureList(const FeatureList &featureList, double roughness, std::string outFilePath, int row, int col)
	{
		int nData = featureList.size();
		int nDim = featureList.at(0)->size();

		std::ofstream ofs(outFilePath);
		ofs << "#[Property]\n";
		ofs << "#DataSize = " << nData << std::endl;
		ofs << "#Dimension = " << nDim << std::endl;
		ofs << "#Roughness = " << roughness << std::endl;
		if (row != 0)
			ofs << "#Row = " << row << std::endl;
		if (col != 0)
			ofs << "#Col = " << col << std::endl;
		ofs << "#[Data]\n";

		for (int i = 0; i < nData; i++)
		{
			for (int j = 0; j < nDim; j++)
			{
				ofs << featureList.at(i)->at(j);
				if (j != nDim - 1)
				{
					ofs << ",";
				}
				else
				{
					ofs << "\n";
				}
			}
		}
		return true;
	}

	/*
	* responseList�̒��g��outFilePath�ɏ����o��
	*/
	bool saveResponseList(const ResponseList &responseList, double roughness, std::string outFilePath, int row, int col)
	{
		int nData = responseList.size();
		int nDim = responseList.at(0)->size();

		std::ofstream ofs(outFilePath);
		ofs << "#[Property]\n";
		ofs << "#DataSize = " << nData << std::endl;
		ofs << "#Dimension = " << nDim << std::endl;
		ofs << "#Roughness = " << roughness << std::endl;
		if (row != 0)
			ofs << "#Row = " << row << std::endl;
		if (col != 0)
			ofs << "#Col = " << col << std::endl;
		ofs << "#[Data]\n";

		for (int i = 0; i < nData; i++)
		{
			for (int j = 0; j < nDim; j++)
			{
				ofs << responseList.at(i)->at(j);
				if (j != nDim - 1)
				{
					ofs << ",";
				}
				else
				{
					ofs << "\n";
				}
			}
		}
		return true;
	}

	/*
	*
	*/
	bool txt2Png(std::string inFilePath, std::string outFilePath)
	{
		std::ifstream ifs(inFilePath);
		std::string str;
		bool bData = false;
		cv::Mat output_image;
		int nRow, nCol;
		int x = 0, y = 0;

		if (ifs.fail())
		{
			std::cout << "File�̓ǂݍ��݃G���[" << std::endl;
			return false;
		}

		while (getline(ifs, str))
		{
			////////////////////////
			////normal�̓ǂݍ���////
			////////////////////////
			if (bData)
			{
				std::string token;
				std::istringstream stream(str);
				int count = 0;
				int zeroCount = 0; //�l���[���̗v�f�̐�
				cv::Vec3d tempNormalVec;
				while (getline(stream, token, ','))
				{
					if (count == 0)
					{
						//nx�������
						output_image.data[y * output_image.step + x * output_image.elemSize() + (2 - count) * output_image.elemSize1()] = (int)((std::stod(token) + 1) * 255 / 2);
						if (std::stod(token) == 0)
						{
							zeroCount++;
						}
					}
					else if (count == 1)
					{
						//ny�������
						output_image.data[y * output_image.step + x * output_image.elemSize() + (2 - count) * output_image.elemSize1()] = (int)((std::stod(token) + 1) * 255 / 2);
						if (std::stod(token) == 0)
						{
							zeroCount++;
						}
					}
					else if (count == 2)
					{
						//nz�������
						output_image.data[y * output_image.step + x * output_image.elemSize() + (2 - count) * output_image.elemSize1()] = (int)((std::stod(token) + 1) * 255 / 2);
						if (std::stod(token) == 0)
						{
							zeroCount++;
						}

						//�v�f�����ׂ�0�̂��̂͘g�O�Ƃ���
						if (zeroCount == 3)
						{
							output_image.data[y * output_image.step + x * output_image.elemSize()] = 0;
							output_image.data[y * output_image.step + x * output_image.elemSize() + 1 * output_image.elemSize1()] = 0;
							output_image.data[y * output_image.step + x * output_image.elemSize() + 2 * output_image.elemSize1()] = 0;
						}
					}
					count++;
				}
				x++;
				y = y + (x / nCol);
				x = x % nCol;
			}
			//////////////////////////
			////Property�̓ǂݍ���////
			//////////////////////////
			else
			{
				UtilityMethod::deleteSpace(str);
				if (str == "#[Data]")
				{
					bData = true;
					output_image = cv::Mat::zeros(nRow, nCol, CV_8UC3);
				}
				std::string token;
				std::istringstream stream(str);
				int count = 0, nCase = 0;
				while (getline(stream, token, '='))
				{
					if (count == 0)
					{
						if (token == "#Row")
							nCase = 1;
						if (token == "#Col")
							nCase = 2;
					}
					if (count == 1)
					{
						if (nCase == 1)
							nRow = std::stoi(token) - 2;
						if (nCase == 2)
							nCol = std::stoi(token) - 2;
					}
					count++;
				}
			}
		}

		cv::imwrite(outFilePath + "_normal.png", output_image);

		//cv::Mat alpha_image = cv::Mat(output_image.size(), CV_8UC3);
		//cv::cvtColor(output_image, alpha_image, CV_RGB2RGBA);
		//for (int y = 0; y < alpha_image.rows; ++y) {
		//	for (int x = 0; x < alpha_image.cols; ++x) {
		//		cv::Vec4b px = alpha_image.at<cv::Vec4b>(y, x);
		//		if (px[0] + px[1] + px[2] == 0) {
		//			px[3] = 0;
		//			alpha_image.at<cv::Vec4b>(y, x) = px;
		//		}
		//	}
		//}

		//cv::imwrite(outFilePath + "_normal.png", alpha_image);

		return true;
	}

	//template<typename Type>
	//bool vectorNormalizing(std::vector<Type>& vec)
	//{
	//	Type sum = 0.0;
	//	for (int i = 0; i < vec.size(); i++)
	//		sum += vec.at(i) * vec.at(i);
	//	for (int i = 0; i < vec.size(); i++)
	//		vec.at(i) = vec.at(i) / sum;
	//	return true;
	//}

	bool vectorNormalizing(Feature& vec)
	{
		double sum = 0.0;
		for (int i = 0; i < vec.size(); i++)
			sum += vec.at(i) * vec.at(i);

		sum = sqrt(sum);

		if (sum == 0.0)
			return false;

		for (int i = 0; i < vec.size(); i++)
			vec.at(i) = vec.at(i) / sum;
		return true;
	}

	/*
	* �s��̗v�f�������ڂ̂܂܂̔z��ŕۑ�����D
	*	?�w�b�_�[����
	*	?��F�X�y�[�X��؂�
	*	?�s�F���s��؂�
	*
	* filename : file��path
	* mat : ��������mat
	*/
	template<typename Type>
	bool writeTxt(const std::string filename, const cv::Mat_<Type> & mat)
	{
		using std::string;
		string line;
		std::ofstream ofs(filename.c_str());
		if (!ofs)
		{
			std::cout << boost::format(" cannot open %s\n") % filename;
			return false;
		}

		for (int j = 0; j<mat.rows; j++)
		{
			for (int i = 0; i<mat.cols; i++)
			{
				ofs << mat(j, i);
				if (i < mat.cols - 1) ofs << " ";
				if (i == mat.cols - 1) ofs << "\n";
			}
		}
		return true;
	}

	/*
	* �s��̗v�f�������ڂ̂܂܂̔z��ŕۑ�����D
	*	?�w�b�_�[����
	*	?��F�X�y�[�X��؂�
	*	?�s�F���s��؂�
	*
	* filename : file��path
	* mat : ��������mat
	*/
	bool writeTxt_float(const std::string filename, const cv::Mat & mat)
	{
		using std::string;
		string line;
		std::ofstream ofs(filename.c_str());
		if (!ofs)
		{
			std::cout << boost::format(" cannot open %s\n") % filename;
			return false;
		}

		for (int j = 0; j<mat.rows; j++)
		{
			for (int i = 0; i<mat.cols; i++)
			{
				ofs << mat.at<float>(j, i);
				if (i < mat.cols - 1) ofs << " ";
				if (i == mat.cols - 1) ofs << "\n";
			}
		}
		return true;
	}


	/*
	* �s��̗v�f�������ڂ̂܂܂̔z��ŕۑ�����D
	*	?�w�b�_�[����
	*	?��F�X�y�[�X��؂�
	*	?�s�F���s��؂�
	*/
	template<typename Type>
	bool readTxt(const std::string filename, cv::Mat_<Type> & mat)
	{
		using std::string;
		string line;
		std::ifstream ifs(filename.c_str());
		if (!ifs)
		{
			std::cout << boost::format(" cannot open %s\n") % filename;
			return false;
		}

		mat = cv::Mat_<Type>();

		while (getline(ifs, line))
		{
			boost::trim(line);
			std::list<std::string> results;
			boost::split(results, line, boost::is_any_of(" \t"),
				boost::token_compress_on);

			cv::Mat_<Type> row(1, results.size());
			std::list<std::string>::iterator iter = results.begin();
			std::list<std::string>::iterator tail = results.end();
			for (int i = 0; iter != tail; ++iter, i++)
			{
				row(i) = boost::lexical_cast<Type>(*iter);
			}

			mat.push_back(row);
		}
		return true;
	};
}
