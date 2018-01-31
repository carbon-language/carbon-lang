cmake_minimum_required(VERSION 3.9.0)
project(f18)

set(GCC /home/sw/thirdparty/gcc/gcc-7.3.0/linux86-64/redhat)
set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH true)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17")

set(SOURCES
  char-buffer.cc
  f2018-demo.cc
  idioms.cc
  message.cc
  parse-tree.cc
  position.cc
  preprocessor.cc
  prescan.cc
  source.cc
)
add_executable(f18 ${SOURCES})

add_executable(type-test type.cc idioms.cc)
