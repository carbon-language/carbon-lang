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
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
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

# Because compiler-rt spends a lot of time setting up custom compile flags,
# define a handy helper function for it. The compile flags setting in CMake
# has serious issues that make its syntax challenging at best.
function(set_target_compile_flags target)
  foreach(arg ${ARGN})
    set(argstring "${argstring} ${arg}")
  endforeach()
  set_property(TARGET ${target} PROPERTY COMPILE_FLAGS "${argstring}")
endfunction()

add_subdirectory(lib)

if(LLVM_INCLUDE_TESTS)
  # Currently the tests have not been ported to CMake, so disable this
  # directory.
  #
  #add_subdirectory(test)
endif()
