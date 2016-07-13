#pragma once
#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdafx.h"
#include <vector>

// Value type that is used for both feature and response. 
// Change "float"->"double" for higher precision (with more computation cost)
typedef float VAL_TYPE;

// Definition of Feature and Response
typedef std::vector<VAL_TYPE> Feature;
typedef std::vector<VAL_TYPE> Response;

// Definition of Feature and Response lists
typedef std::vector<const Feature*> FeatureList;
typedef std::vector<const Response*> ResponseList;

// Ammae
//typedef std::vector<int> Index;						//各クエリのノーマルのインデックス
typedef std::vector<double> IndexDist;					//各クエリのノーマルとの距離
//typedef std::vector<Index*> IndexList;				//全てのクエリのノーマルのインデックス
typedef std::vector<IndexDist*> IndexDistList;			//全てのクエリのノーマルとの距離
typedef std::vector<ResponseList*> KnnList;				//全てのクエリについてクエリ毎のKNN結果のノーマル


// other type defs 
typedef unsigned int uint;

const std::string GLOBAL_HOME_DIR = "D:/Data/PhotometricStereo/";
const int GLOBAL_WINDOW_SIZE = 1;
const int GLOBAL_SIGMA_TYPE = 5;									//Sigmaを用意する数

const int GLOBAL_KNN_COUNT = 30;									//knnのkをいくつに設定するか
const int GLOBAL_MRF_MAX = 2;										//MRFの操作を何回するか(default : 6)
const int GLOBAL_MRF_ITERATION = 30;								//MRFの操作一回で何回繰り返しを行うか(default : 30)

const std::size_t GLOBAL_VEC_LIMIT = 10000000;						//vectorを初期化するときに要素を確保する量(ピクセル数(=パッチ数)に相当)
const int GLOBAL_LIGHT_MAX = 100;									//光ベクトルの数の最大値
const int GLOBAL_PIC_MAX_SIZE = 9999;								//一つの入力画像の最大width,height

#endif