cmake_minimum_required(VERSION 3.9.0)

# Pass -DGCC=... to cmake to use a specific gcc installation.
if( GCC )
  set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
  set(CMAKE_CC_COMPILER "${GCC}/bin/gcc")
  set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH true)
endif()
if(CLANG)
  set(CMAKE_CXX_COMPILER "${CLANG}/bin/clang")
  set(CMAKE_CC_COMPILER "${CLANG}/bin/clang")
  if(GCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
  endif()
endif()

# Reminder: Setting CMAKE_CXX_COMPILER must be done before calling project()

project(f18 CXX)

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
     if(GCC)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
     endif()
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
