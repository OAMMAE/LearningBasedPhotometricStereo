#pragma once

#include "stdafx.h"
#include "utility.h"

namespace UtilityMethod
{

#pragma region Quick Sort

	/*
	* 軸要素の選択
	* 順に見て、最初に見つかった異なる2つの要素のうち、
	* 大きいほうの番号を返します。
	* 全部同じ要素の場合は -1 を返します。
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
	* パーティション分割
	* Data[i]～Data[j]の間で、x を軸として分割します。
	* x より小さい要素は前に、大きい要素はうしろに来ます。
	* 大きい要素の開始番号を返します。
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
	* クイックソート（再帰用）
	* 配列Dataの、Data[i]からData[j]を並べ替えます。
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
	* folderNameという名前のフォルダを作る
	*/
	void mkdir(std::string folderName)
	{
		//string から char* への変換
		int len = folderName.length();
		char* fname = new char[len + 1];
		memcpy(fname, folderName.c_str(), len + 1);

		_mkdir(fname);
		delete fname;

		return;
	}

	/*
	* stringという文字列に対して、末尾の0を消去する
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