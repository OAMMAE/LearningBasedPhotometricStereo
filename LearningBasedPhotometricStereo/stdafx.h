// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#define _USE_MATH_DEFINES

#ifdef _MSC_VER 
#include "targetver.h"
#endif

///////////////////////////////////////
//////////自作ヘッダファイル///////////
///////////////////////////////////////
#ifdef _MSC_VER 
#include "opencv_library_controller.h"
#endif

///////////////////////////////////////
//////////標準ヘッダファイル///////////
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
///////ライブラリヘッダファイル////////
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

// TODO: プログラムに必要な追加ヘッダーをここで参照してください
