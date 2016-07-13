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
//typedef std::vector<int> Index;						//�e�N�G���̃m�[�}���̃C���f�b�N�X
typedef std::vector<double> IndexDist;					//�e�N�G���̃m�[�}���Ƃ̋���
//typedef std::vector<Index*> IndexList;				//�S�ẴN�G���̃m�[�}���̃C���f�b�N�X
typedef std::vector<IndexDist*> IndexDistList;			//�S�ẴN�G���̃m�[�}���Ƃ̋���
typedef std::vector<ResponseList*> KnnList;				//�S�ẴN�G���ɂ��ăN�G������KNN���ʂ̃m�[�}��


// other type defs 
typedef unsigned int uint;

const std::string GLOBAL_HOME_DIR = "D:/Data/PhotometricStereo/";
const int GLOBAL_WINDOW_SIZE = 1;
const int GLOBAL_SIGMA_TYPE = 5;									//Sigma��p�ӂ��鐔

const int GLOBAL_KNN_COUNT = 30;									//knn��k�������ɐݒ肷�邩
const int GLOBAL_MRF_MAX = 2;										//MRF�̑�������񂷂邩(default : 6)
const int GLOBAL_MRF_ITERATION = 30;								//MRF�̑�����ŉ���J��Ԃ����s����(default : 30)

const std::size_t GLOBAL_VEC_LIMIT = 10000000;						//vector������������Ƃ��ɗv�f���m�ۂ����(�s�N�Z����(=�p�b�`��)�ɑ���)
const int GLOBAL_LIGHT_MAX = 100;									//���x�N�g���̐��̍ő�l
const int GLOBAL_PIC_MAX_SIZE = 9999;								//��̓��͉摜�̍ő�width,height

#endif