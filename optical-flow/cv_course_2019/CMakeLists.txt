cmake_minimum_required(VERSION 2.8)

project( ans2_gmg )
find_package( OpenCV REQUIRED )
add_executable( ans2_gmg ans2_gmg.cpp )
target_link_libraries( ans2_gmg ${OpenCV_LIBS} )

project( ans2_mog2 )
find_package( OpenCV REQUIRED )
add_executable( ans2_mog2 ans2_mog2.cpp )
target_link_libraries( ans2_mog2 ${OpenCV_LIBS} )

project( ans2_mog )
find_package( OpenCV REQUIRED )
add_executable( ans2_mog ans2_mog.cpp )
target_link_libraries( ans2_mog ${OpenCV_LIBS} )
