cmake_minimum_required(VERSION 2.8)

project( exc1 )
find_package( OpenCV REQUIRED )
add_executable( exc1 exc1.cpp )
target_link_libraries( exc1 ${OpenCV_LIBS} )

project( exc2 )
find_package( OpenCV REQUIRED )
add_executable( exc2 exc2.cpp )
target_link_libraries( exc2 ${OpenCV_LIBS} )


