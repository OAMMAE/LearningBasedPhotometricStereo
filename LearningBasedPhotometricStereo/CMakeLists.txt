cmake_minimum_required(VERSION 2.8)

project(hello_world)
message("Project name : ${PROJECT_NAME}")

#set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wconversion -Wdisabled-optimization -Wendif-labels -Wfloat-equal -Winit-self -Winline -Wlogical-op -Wmissing-include-dirs -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wpacked -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-promo -Wswitch-default -Wswitch-enum -Wunsafe-loop-optimizations -Wvariadic-macros -Wwrite-strings ")
#set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg")
#set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native")

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE Release)

#find_package( OpenMP )
#if (OPENMP_FOUND) 
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    message ("openmp founded")
#endif()

set(EXE_NAME helloworld)

find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message ("OPENMP found")
endif()

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

message("Ceres include dirs : ${CERES_INCLUDE_DIRS}")
message("Ceres include dirs : ${CERES_INCLUDES}")
message("Ceres library dirs : ${CERES_LIBRARIES}")
message("Ceres library dirs : ${CERES_LIBS}")

find_package(OpenCV 2.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message("OpenCV include dirs : ${OpenCV_INCLUDE_DIRS}")
message("OpenCV include dirs : ${OpenCV_INCLUDES}")
message("OpenCV library dirs : ${OpenCV_LIBRARIES}")
message("OpenCV library dirs : ${OpenCV_LIBS}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

message("Eigen include dirs : ${EIGEN_INCLUDE_DIRS}")
message("Eigen include dirs : ${EIGEN_INCLUDES}")
message("Eigen library dirs : ${EIGEN_LIBRARIES}")
message("Eigen library dirs : ${EIGEN_LIBS}")

add_executable(${EXE_NAME} main.cpp learning_based_photometric_stereo.cpp utility.cpp utility_photometric_stereo.cpp)
target_link_libraries(${EXE_NAME} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})

message("CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

#add_definition ("-std=c++11")

