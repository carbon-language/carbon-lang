cmake_minimum_required(VERSION 3.9.0)

#
# Reminder: The normal way to set the compiler is via the CXX environment variable
#
#   CXX=/opt/gcc-7.3.0/bin/g++ cmake .../f18
#
# but for convenience we provide the following cmake variables to
# use an existing gcc installation directory.
#

# Pass '-DPGI=1' to cmake to use the latest gcc installation at PGI.
# Add '-DCLANG=1' to use clang and libc++.
if( PGI )
  set(GCC /home/sw/thirdparty/gcc/gcc-7.3.0/linux86-64/redhat)
  if(CLANG)
    set(CLANG /home/sw/thirdparty/llvm/linux86-64/6.0-pgi2)
  endif()
endif()

# Pass -DGCC=... to cmake to use a specific gcc installation
if( GCC )
  if (CLANG)
    set(CMAKE_CXX_COMPILER "${CLANG}/bin/clang++")
    set(CMAKE_CC_COMPILER "${CLANG}/bin/clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  else()
    set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
    set(CMAKE_CC_COMPILER "${GCC}/bin/gcc")
  endif()
  set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH true)
endif()

# Reminder: Setting CMAKE_CXX_COMPILER must be done before calling project()

project(f18 CXX)

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   if(CLANG)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
   endif()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
   set(CMAKE_CXX_FLAGS_RELEASE    "-O2 -DDEBUG")
   set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 '-DCHECK=(void)'")
   set(CMAKE_CXX_FLAGS_DEBUG      "-g -DDEBUG")
endif()

set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "FLANG version: ${FLANG_VERSION}")

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories(BEFORE
  ${FLANG_BINARY_DIR}/include
  ${FLANG_SOURCE_DIR}/include
  )

add_subdirectory(include/flang)
add_subdirectory(lib)
add_subdirectory(tools)

configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h)
