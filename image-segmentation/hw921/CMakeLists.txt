cmake_minimum_required(VERSION 2.8)

project( bs1 )
find_package( OpenCV REQUIRED )
add_executable( bs1 bs1.cpp )
target_link_libraries( bs1 ${OpenCV_LIBS} )

project( bs2 )
find_package( OpenCV REQUIRED )
add_executable( bs2 bs2.cpp )
target_link_libraries( bs2 ${OpenCV_LIBS} )

project( _bs1 )
find_package( OpenCV REQUIRED )
add_executable( _bs1 bs1_.cpp )
target_link_libraries( _bs1 ${OpenCV_LIBS} )

project( con )
find_package( OpenCV REQUIRED )
add_executable( con con.cpp )
target_link_libraries( con ${OpenCV_LIBS} )

