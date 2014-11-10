# Toolchain config for Android standalone NDK.
#
# Usage:
# build host llvm and clang first
# cmake -DCMAKE_TOOLCHAIN_FILE=../lldb/cmake/platforms/Android.cmake \
#       -DANDROID_TOOLCHAIN_DIR=<toolchain_dir> \
#       -DANDROID_ABI=<target_abi> \
#       -DCMAKE_CXX_COMPILER_VERSION=<gcc_version> \
#       -DLLVM_TARGET_ARCH=<llvm_target_arch> \
#       -DLLVM_TARGETS_TO_BUILD=<llvm_targets_to_build> \
#       -DLLVM_TABLEGEN=<path_to_llvm-tblgen> \
#       -DCLANG_TABLEGEN=<path_to_clang-tblgen>
#
# Current Support:
#   ANDROID_ABI = x86, x86_64
#   CMAKE_CXX_COMPILER_VERSION = 4.9
#   LLVM_TARGET_ARCH = X86
#   LLVM_TARGETS_TO_BUILD = X86
#   LLVM_TABLEGEN = path to host llvm-tblgen
#   CLANG_TABLEGEN = path to host clang-tblgen

if( DEFINED CMAKE_CROSSCOMPILING )
 return()
endif()

get_property( IS_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE )
if( IS_IN_TRY_COMPILE )
 # this seems necessary and works fine but I'm unsure if it breaks anything
 return()
endif()

set( CMAKE_SYSTEM_NAME Linux )
include( CMakeForceCompiler )

# flags and definitions
remove_definitions( -DANDROID -D__ANDROID__ )
add_definitions( -DANDROID -D__ANDROID_NDK__ -DLLDB_DISABLE_LIBEDIT )
set( ANDROID True )
set( __ANDROID_NDK__ True )

set( ANDROID_ABI "${ANDROID_ABI}" CACHE INTERNAL "Android Abi" FORCE )
if( ANDROID_ABI STREQUAL "x86" )
 set( CMAKE_SYSTEM_PROCESSOR "i686" )
 set( ANDROID_TOOLCHAIN_NAME "x86-linux-android" )
elseif( ANDROID_ABI STREQUAL "x86_64" )
 set( CMAKE_SYSTEM_PROCESSOR "x86_64" )
 set( ANDROID_TOOLCHAIN_NAME "x86_64-linux-android" )
else()
 message( SEND_ERROR "Unknown ANDROID_ABI = \"${ANDROID_ABI}\"." )
endif()

set( ANDROID_TOOLCHAIN_DIR "${ANDROID_TOOLCHAIN_DIR}" CACHE INTERNAL "Android standalone toolchain directory" FORCE )
set( ANDROID_SYSROOT "${ANDROID_TOOLCHAIN_DIR}/sysroot" CACHE INTERNAL "Android Sysroot" FORCE )

# force python exe to be the one in Android toolchian
set( PYTHON_EXECUTABLE "${ANDROID_TOOLCHAIN_DIR}/bin/python" CACHE INTERNAL "Python exec path" FORCE )

if( NOT CMAKE_C_COMPILER )
 set( CMAKE_C_COMPILER   "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-gcc"     CACHE PATH "C compiler" )
 set( CMAKE_CXX_COMPILER "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-g++"     CACHE PATH "C++ compiler" )
 set( CMAKE_ASM_COMPILER "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-gcc"     CACHE PATH "assembler" )
 set( CMAKE_STRIP        "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-strip"   CACHE PATH "strip" )
 set( CMAKE_AR           "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-ar"      CACHE PATH "archive" )
 set( CMAKE_LINKER       "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-ld"      CACHE PATH "linker" )
 set( CMAKE_NM           "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-nm"      CACHE PATH "nm" )
 set( CMAKE_OBJCOPY      "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-objcopy" CACHE PATH "objcopy" )
 set( CMAKE_OBJDUMP      "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-objdump" CACHE PATH "objdump" )
 set( CMAKE_RANLIB       "${ANDROID_TOOLCHAIN_DIR}/bin/${ANDROID_TOOLCHAIN_NAME}-ranlib"  CACHE PATH "ranlib" )
endif()

set( ANDROID_CXX_FLAGS "--sysroot=${ANDROID_SYSROOT} -pie -fPIE -funwind-tables -fsigned-char -no-canonical-prefixes" )
# TODO: different ARM abi have different flags such as neon, vfpv etc
if( X86 )
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -funswitch-loops -finline-limit=300" )
endif()

# linker flags
set( ANDROID_CXX_FLAGS    "${ANDROID_CXX_FLAGS} -fdata-sections -ffunction-sections" )
set( ANDROID_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} -Wl,--gc-sections" )

# cache flags
set( CMAKE_CXX_FLAGS           ""                        CACHE STRING "c++ flags" )
set( CMAKE_C_FLAGS             ""                        CACHE STRING "c flags" )
set( CMAKE_EXE_LINKER_FLAGS    "-Wl,-z,nocopyreloc"      CACHE STRING "executable linker flags" )
set( ANDROID_CXX_FLAGS         "${ANDROID_CXX_FLAGS}"    CACHE INTERNAL "Android c/c++ flags" )
set( ANDROID_LINKER_FLAGS      "${ANDROID_LINKER_FLAGS}" CACHE INTERNAL "Android c/c++ linker flags" )

# final flags
set( CMAKE_CXX_FLAGS           "${ANDROID_CXX_FLAGS} ${CMAKE_CXX_FLAGS}" )
set( CMAKE_C_FLAGS             "${ANDROID_CXX_FLAGS} ${CMAKE_C_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS    "${ANDROID_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}" )

# global includes and link directories
set( ANDROID_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_DIR}/include/c++/${ANDROID_COMPILER_VERSION}" )
list( APPEND ANDROID_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_DIR}/include/python2.7" )
include_directories( SYSTEM "${ANDROID_SYSROOT}/usr/include" ${ANDROID_INCLUDE_DIRS} )

# target environment
set( CMAKE_FIND_ROOT_PATH "${ANDROID_TOOLCHAIN_DIR}/bin" "${ANDROID_TOOLCHAIN_DIR}/${ANDROID_TOOLCHAIN_NAME}" "${ANDROID_SYSROOT}" )

# only search for libraries and includes in the ndk toolchain
set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )