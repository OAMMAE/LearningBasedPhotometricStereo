#pragma once

#include "stdafx.h"
#include "utility.h"

namespace UtilityMethod
{

#pragma region Quick Sort

	/*
	* ���v�f�̑I��
	* ���Ɍ��āA�ŏ��Ɍ��������قȂ�2�̗v�f�̂����A
	* �傫���ق��̔ԍ���Ԃ��܂��B
	* �S�������v�f�̏ꍇ�� -1 ��Ԃ��܂��B
	*/
	int pivot(std::vector<double> Data, int i, int j)
	{
		int k = i + 1;
		while (k <= j && Data.at(i) == Data.at(k)) k++;
		if (k > j) return -1;
		if (Data.at(i) >= Data.at(k)) return i;
		return k;
	}

	/*
	* �p�[�e�B�V��������
	* Data[i]�`Data[j]�̊ԂŁAx �����Ƃ��ĕ������܂��B
	* x ��菬�����v�f�͑O�ɁA�傫���v�f�͂�����ɗ��܂��B
	* �傫���v�f�̊J�n�ԍ���Ԃ��܂��B
	*/
	int partition(std::vector<int> &Index, std::vector<double> &Data, int i, int j, double x)
	{
		int l = i, r = j;

		while (l <= r)
		{
			while (l <= j && Data.at(l) < x) l++;

			while (r >= i && Data.at(r) >= x) r--;

			if (l > r) break;
			double tempData = Data.at(l);
			int tempIndex = Index.at(l);
			Data.at(l) = Data.at(r);
			Data.at(r) = tempData;
			Index.at(l) = Index.at(r);
			Index.at(r) = tempIndex;

			l++; r--;
		}
		return l;
	}

	/*
	* �N�C�b�N�\�[�g�i�ċA�p�j
	* �z��Data�́AData[i]����Data[j]����בւ��܂��B
	*/
	void quickSort(std::vector<int> &Index, std::vector<double> &Data, int i, int j)
	{
		if (i >= j)
		{
			return;
		}
		int p = pivot(Data, i, j);
		if (p != -1)
		{
			int k = partition(Index, Data, i, j, Data.at(p));
			quickSort(Index, Data, i, k - 1);
			quickSort(Index, Data, k, j);
		}
	}

#pragma endregion

	/*
	* folderName�Ƃ������O�̃t�H���_�����
	*/
	void mkdir(std::string folderName)
	{
		//string ���� char* �ւ̕ϊ�
		int len = folderName.length();
		char* fname = new char[len + 1];
		memcpy(fname, folderName.c_str(), len + 1);

		_mkdir(fname);
		delete fname;

		return;
	}

	/*
	* string�Ƃ���������ɑ΂��āA������0����������
	*/
	void deleteSuffix0(std::string &string)
	{
		int nStringLength = string.size();
		for (int i = 0; i < nStringLength; i++)
		{
			char cTempChar = string.back();
			if (cTempChar == '0')
			{
				string.pop_back();
			}
			else
			{
				break;
			}
		}
	}
}