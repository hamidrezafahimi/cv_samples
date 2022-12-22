cmake_minimum_required(VERSION 2.8.12)

PROJECT(license_generator)

set (CMAKE_CXX_STANDARD 11)

find_package( OpenCV 4.1.1 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} src/${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(license_generator)
