cmake_minimum_required(VERSION 3.9.0)

set(GCC /home/sw/thirdparty/gcc/gcc-7.3.0/linux86-64/redhat)
set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH true)

project(f18)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17")

set(SOURCES
  tools/f18/f18.cc
  lib/parser/char-buffer.cc
  lib/parser/idioms.cc
  lib/parser/message.cc
  lib/parser/parse-tree.cc
  lib/parser/position.cc
  lib/parser/preprocessor.cc
  lib/parser/prescan.cc
  lib/parser/source.cc
)
add_executable(f18 ${SOURCES})
add_executable(type-test type.cc idioms.cc attr.cc)
