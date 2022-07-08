cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)

project( project )
find_package( OpenCV REQUIRED )
add_executable( project project.cpp )
target_link_libraries( project ${OpenCV_LIBS} )

project( adjustment )
find_package( OpenCV REQUIRED )
add_executable( adjustment adjustment.cpp )
target_link_libraries( adjustment ${OpenCV_LIBS} )

project( detection )
find_package( OpenCV REQUIRED )
add_executable( detection detection.cpp )
target_link_libraries( detection ${OpenCV_LIBS} )

project( equal )
find_package( OpenCV REQUIRED )
add_executable( equal equal.cpp )
target_link_libraries( equal ${OpenCV_LIBS} )
find_package( Tesseract 3.05 REQUIRED ) # 3.05 is currently the latest version of the git repository.
include_directories(${Tesseract_INCLUDE_DIRS})
target_link_libraries(<your_program_executable> ${Tesseract_LIBRARIES}) 
