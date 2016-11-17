#pragma once

#include "stdafx.h"

namespace UtilityMethod
{
	int pivot(std::vector<double> Data, int i, int j);
	int partition(std::vector<int> &Index, std::vector<double> &Data, int i, int j, double x);
	void quickSort(std::vector<int> &Index, std::vector<double> &Data, int i, int j);

	void str_mkdir(std::string folderName);
	void deleteSuffix0(std::string &string);


	/*
	* 空白(スペース，タブ)を削除
	* @param[inout] buf 処理文字列
	*/
	inline void deleteSpace(std::string &buf)
	{
		size_t pos;
		while ((pos = buf.find_first_of("  \t")) != std::string::npos) {
			buf.erase(pos, 1);
		}
	}
}