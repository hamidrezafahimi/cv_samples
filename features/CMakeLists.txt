cmake_minimum_required(VERSION 2.8)

project( sift )
find_package( OpenCV REQUIRED )
add_executable( sift sift/MySIFT.cpp )
target_link_libraries( sift ${OpenCV_LIBS} )


