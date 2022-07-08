cmake_minimum_required(VERSION 2.8)

project( hou )
find_package( OpenCV REQUIRED )
add_executable( hou hou.cpp )
target_link_libraries( hou ${OpenCV_LIBS} )

project( thr )
find_package( OpenCV REQUIRED )
add_executable( thr thr.cpp )
target_link_libraries( thr ${OpenCV_LIBS} )


