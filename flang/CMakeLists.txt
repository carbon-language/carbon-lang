cmake_minimum_required(VERSION 3.9.0)

set(GCC /home/sw/thirdparty/gcc/gcc-7.3.0/linux86-64/redhat)
set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
set(CMAKE_BUILD_WITH_INSTALL_RPATH true)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -std=c++17")

set(SOURCES_F18
  tools/f18/f18.cc
  lib/parser/char-buffer.cc
  lib/parser/idioms.cc
  lib/parser/message.cc
  lib/parser/parse-tree.cc
  lib/parser/preprocessor.cc
  lib/parser/prescan.cc
  lib/parser/provenance.cc
  lib/parser/source.cc
  lib/parser/token-sequence.cc
)
add_executable(f18 ${SOURCES_F18})

set(SOURCES_TEST_TYPE
  tools/f18/test-type.cc
  lib/semantics/type.cc
  lib/semantics/attr.cc
  lib/parser/char-buffer.cc
  lib/parser/idioms.cc
  lib/parser/message.cc
  lib/parser/parse-tree.cc
  lib/parser/position.cc
  lib/parser/preprocessor.cc
  lib/parser/prescan.cc
  lib/parser/source.cc
)
add_executable(test-type ${SOURCES_TEST_TYPE})
