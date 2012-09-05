# CMake build for CompilerRT.
#
# This build assumes that CompilerRT is checked out into the
# 'projects/compiler-rt' inside of an LLVM tree, it is not a stand-alone build
# system.
#
# An important constraint of the build is that it only produces libraries
# based on the ability of the host toolchain to target various platforms.

include(LLVMParseArguments)

# The CompilerRT build system requires CMake version 2.8.8 or higher in order
# to use its support for building convenience "libraries" as a collection of
# .o files. This is particularly useful in producing larger, more complex
# runtime libraries.
cmake_minimum_required(VERSION 2.8.8)

# FIXME: Below we assume that the target build of LLVM/Clang is x86, which is
# not at all valid. Much of this can be fixed just by switching to use
# a just-built-clang binary for the compiles.

# Detect whether the current target platform is 32-bit or 64-bit, and setup
# the correct commandline flags needed to attempt to target 32-bit and 64-bit.
if(CMAKE_SIZEOF_VOID_P EQUAL 4 OR LLVM_BUILD_32_BITS)
  set(TARGET_X86_64_CFLAGS "-m64")
  set(TARGET_I386_CFLAGS "")
else()
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Please use a sane architecture with 4 or 8 byte pointers.")
  endif()
  set(TARGET_X86_64_CFLAGS "")
  set(TARGET_I386_CFLAGS "-m32")
endif()

# Try to compile a very simple source file to ensure we can target the given
# platform. We use the results of these tests to build only the various target
# runtime libraries supported by our current compilers cross-compiling
# abilities.
set(SIMPLE_SOURCE64 ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple64.c)
file(WRITE ${SIMPLE_SOURCE64} "#include <stdlib.h>\nint main() {}")
try_compile(CAN_TARGET_X86_64 ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE64}
            COMPILE_DEFINITIONS "${TARGET_X86_64_CFLAGS}"
            CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${TARGET_X86_64_CFLAGS}")

set(SIMPLE_SOURCE32 ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple32.c)
file(WRITE ${SIMPLE_SOURCE32} "#include <stdlib.h>\nint main() {}")
try_compile(CAN_TARGET_I386 ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE32}
            COMPILE_DEFINITIONS "${TARGET_I386_CFLAGS}"
            CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${TARGET_I386_CFLAGS}")

if(LLVM_ANDROID_TOOLCHAIN_DIR)
  if(EXISTS ${LLVM_ANDROID_TOOLCHAIN_DIR}/arm-linux-androideabi)
    set(CAN_TARGET_ARM_ANDROID 1)
    set(TARGET_ARM_ANDROID_CFLAGS
      -target arm-linux-androideabi
      --sysroot=${LLVM_ANDROID_TOOLCHAIN_DIR}/sysroot
      -B${LLVM_ANDROID_TOOLCHAIN_DIR}
      )
  else()
    set(CAN_TARGET_ARM_ANDROID 0)
  endif()
  # TODO: support i686 and MIPS Android toolchains
endif()

function(filter_available_targets out_var)
  set(archs)
  foreach(arch ${ARGN})
    if(${arch} STREQUAL "x86_64" AND CAN_TARGET_X86_64)
      list(APPEND archs ${arch})
    elseif (${arch} STREQUAL "i386" AND CAN_TARGET_I386)
      list(APPEND archs ${arch})
    endif()
  endforeach()
  set(${out_var} ${archs} PARENT_SCOPE)
endfunction()

# Provide some common commmandline flags for Sanitizer runtimes.
set(SANITIZER_COMMON_CFLAGS
  -fPIC
  -fno-builtin
  -fno-exceptions
  -fomit-frame-pointer
  -funwind-tables
  -fvisibility=hidden
  -O3
  )
if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-variadic-macros)
endif()
check_cxx_compiler_flag(-Wno-c99-extensions
  SUPPORTS_NO_C99_EXTENSIONS_FLAG)
if(SUPPORTS_NO_C99_EXTENSIONS_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-c99-extensions)
endif()
if(APPLE)
  list(APPEND SANITIZER_COMMON_CFLAGS -mmacosx-version-min=10.5)
endif()

# Because compiler-rt spends a lot of time setting up custom compile flags,
# define a handy helper function for it. The compile flags setting in CMake
# has serious issues that make its syntax challenging at best.
function(set_target_compile_flags target)
  foreach(arg ${ARGN})
    set(argstring "${argstring} ${arg}")
  endforeach()
  set_property(TARGET ${target} PROPERTY COMPILE_FLAGS "${argstring}")
endfunction()

# Add the public header's directory to the includes for all of compiler-rt.
include_directories(include)

# Build utils before building compiler-rt library.
add_subdirectory(utils)

add_subdirectory(lib)

if(LLVM_INCLUDE_TESTS)
  # Currently the tests have not been ported to CMake, so disable this
  # directory.
  #
  #add_subdirectory(test)
endif()
