// stdafx.h : �W���̃V�X�e�� �C���N���[�h �t�@�C���̃C���N���[�h �t�@�C���A�܂���
// �Q�Ɖ񐔂������A�����܂�ύX����Ȃ��A�v���W�F�N�g��p�̃C���N���[�h �t�@�C��
// ���L�q���܂��B
//

#pragma once

#define _USE_MATH_DEFINES

#ifdef _MSC_VER 
#include "targetver.h"
#endif

///////////////////////////////////////
//////////����w�b�_�t�@�C��///////////
///////////////////////////////////////
#ifdef _MSC_VER 
#include "opencv_library_controller.h"
#endif

///////////////////////////////////////
//////////�W���w�b�_�t�@�C��///////////
///////////////////////////////////////
#include <stdio.h>

#ifdef _MSC_VER 
#include <tchar.h>
#endif

#include <stdlib.h>
#include <time.h>
#include <new>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#ifdef _MSC_VER 
#include <direct.h> // _mkdir for windows
#else
#include <sys/stat.h> // for os x
#endif

#include <vector>
#include <array>

#include <math.h>

///////////////////////////////////////
///////���C�u�����w�b�_�t�@�C��////////
///////////////////////////////////////
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/progress.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

// TODO: �v���O�����ɕK�v�Ȓǉ��w�b�_�[�������ŎQ�Ƃ��Ă�������
