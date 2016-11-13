// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#define _USE_MATH_DEFINES

#include "targetver.h"

///////////////////////////////////////
//////////自作ヘッダファイル///////////
///////////////////////////////////////
#include "opencv_library_controller.h"
#include "common.h"

///////////////////////////////////////
//////////標準ヘッダファイル///////////
///////////////////////////////////////
#include <stdio.h>
#include <tchar.h>

#include <stdlib.h>
#include <time.h>
#include <new>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <direct.h>
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
