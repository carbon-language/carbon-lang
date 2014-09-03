# Toolchain config for iOS.
#
# Usage:
# mkdir build; cd build
# cmake ..; make
# mkdir ios; cd ios
# cmake -DLLVM_IOS_TOOLCHAIN_DIR=/path/to/ios/ndk \
#   -DCMAKE_TOOLCHAIN_FILE=../../cmake/platforms/iOS.cmake ../..
# make <target>

SET(CMAKE_SYSTEM_NAME Darwin)
SET(CMAKE_SYSTEM_VERSION 13)
SET(CMAKE_CXX_COMPILER_WORKS True)
SET(CMAKE_C_COMPILER_WORKS True)
SET(DARWIN_TARGET_OS_NAME ios)

IF(NOT DEFINED ENV{SDKROOT})
 MESSAGE(FATAL_ERROR "SDKROOT env var must be set: " $ENV{SDKROOT})
ENDIF()

IF(NOT CMAKE_C_COMPILER)
  execute_process(COMMAND xcrun -sdk iphoneos -find clang
   OUTPUT_VARIABLE CMAKE_C_COMPILER
   ERROR_QUIET
   OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using c compiler ${CMAKE_C_COMPILER}")
ENDIF()

IF(NOT CMAKE_CXX_COMPILER)
  execute_process(COMMAND xcrun -sdk iphoneos -find clang++
   OUTPUT_VARIABLE CMAKE_CXX_COMPILER
   ERROR_QUIET
   OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using c compiler ${CMAKE_CXX_COMPILER}")
ENDIF()

IF (NOT DEFINED IOS_MIN_TARGET)
execute_process(COMMAND xcodebuild -sdk iphoneos -version SDKVersion
   OUTPUT_VARIABLE IOS_MIN_TARGET
   ERROR_QUIET
   OUTPUT_STRIP_TRAILING_WHITESPACE)
ENDIF()

SET(IOS_COMMON_FLAGS "-isysroot $ENV{SDKROOT} -mios-version-min=${IOS_MIN_TARGET}")
SET(CMAKE_C_FLAGS "${IOS_COMMON_FLAGS}" CACHE STRING "toolchain_cflags" FORCE)
SET(CMAKE_CXX_FLAGS "${IOS_COMMON_FLAGS}" CACHE STRING "toolchain_cxxflags" FORCE)
SET(CMAKE_LINK_FLAGS "${IOS_COMMON_FLAGS}" CACHE STRING "toolchain_linkflags" FORCE)
