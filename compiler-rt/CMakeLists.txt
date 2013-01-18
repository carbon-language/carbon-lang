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

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )
include(AddCompilerRT)

set(COMPILER_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Detect whether the current target platform is 32-bit or 64-bit, and setup
# the correct commandline flags needed to attempt to target 32-bit and 64-bit.
if(CMAKE_SIZEOF_VOID_P EQUAL 4 OR LLVM_BUILD_32_BITS)
  set(TARGET_64_BIT_CFLAGS "-m64")
  set(TARGET_32_BIT_CFLAGS "")
else()
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Please use a sane architecture with 4 or 8 byte pointers.")
  endif()
  set(TARGET_64_BIT_CFLAGS "")
  set(TARGET_32_BIT_CFLAGS "-m32")
endif()

# FIXME: Below we assume that the target build of LLVM/Clang is x86, which is
# not at all valid. Much of this can be fixed just by switching to use
# a just-built-clang binary for the compiles.

set(TARGET_x86_64_CFLAGS ${TARGET_64_BIT_CFLAGS})
set(TARGET_i386_CFLAGS ${TARGET_32_BIT_CFLAGS})

set(COMPILER_RT_SUPPORTED_ARCH
  x86_64 i386)

function(get_target_flags_for_arch arch out_var)
  list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
  if(ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "Unsupported architecture: ${arch}")
  else()
    set(${out_var} ${TARGET_${arch}_CFLAGS} PARENT_SCOPE)
  endif()
endfunction()

# Try to compile a very simple source file to ensure we can target the given
# platform. We use the results of these tests to build only the various target
# runtime libraries supported by our current compilers cross-compiling
# abilities.
set(SIMPLE_SOURCE64 ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple64.c)
file(WRITE ${SIMPLE_SOURCE64} "#include <stdlib.h>\nint main() {}")
try_compile(CAN_TARGET_x86_64 ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE64}
            COMPILE_DEFINITIONS "${TARGET_x86_64_CFLAGS}"
            CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${TARGET_x86_64_CFLAGS}")

set(SIMPLE_SOURCE32 ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple32.c)
file(WRITE ${SIMPLE_SOURCE32} "#include <stdlib.h>\nint main() {}")
try_compile(CAN_TARGET_i386 ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE32}
            COMPILE_DEFINITIONS "${TARGET_i386_CFLAGS}"
            CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${TARGET_i386_CFLAGS}")

# We only support running instrumented tests when we're not cross compiling
# and target a unix-like system. On Android we define the rules for building
# unit tests, but don't execute them.
if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX AND NOT ANDROID)
  set(COMPILER_RT_CAN_EXECUTE_TESTS TRUE)
else()
  set(COMPILER_RT_CAN_EXECUTE_TESTS FALSE)
endif()
    
function(filter_available_targets out_var)
  set(archs)
  foreach(arch ${ARGN})
    list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
    if(NOT (ARCH_INDEX EQUAL -1) AND CAN_TARGET_${arch})
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
  -O3
  )
if(NOT WIN32)
  list(APPEND SANITIZER_COMMON_CFLAGS -fvisibility=hidden)
endif()
# Build sanitizer runtimes with debug info.
check_cxx_compiler_flag(-gline-tables-only SUPPORTS_GLINE_TABLES_ONLY_FLAG)
if(SUPPORTS_GLINE_TABLES_ONLY_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -gline-tables-only)
else()
  list(APPEND SANITIZER_COMMON_CFLAGS -g)
endif()
# Warnings suppressions.
check_cxx_compiler_flag(-Wno-variadic-macros SUPPORTS_NO_VARIADIC_MACROS_FLAG)
if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-variadic-macros)
endif()
check_cxx_compiler_flag(-Wno-c99-extensions SUPPORTS_NO_C99_EXTENSIONS_FLAG)
if(SUPPORTS_NO_C99_EXTENSIONS_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-c99-extensions)
endif()
if(APPLE)
  list(APPEND SANITIZER_COMMON_CFLAGS -mmacosx-version-min=10.5)
endif()

# Architectures supported by Sanitizer runtimes. Specific sanitizers may
# support only subset of these (e.g. TSan works on x86_64 only).
filter_available_targets(SANITIZER_COMMON_SUPPORTED_ARCH
  x86_64 i386)

# Compute the Clang version from the LLVM version.
# FIXME: We should be able to reuse CLANG_VERSION variable calculated
#        in Clang cmake files, instead of copying the rules here.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
       ${PACKAGE_VERSION})
# Setup the paths where compiler-rt runtimes and headers should be stored.
set(LIBCLANG_INSTALL_PATH lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION})
string(TOLOWER ${CMAKE_SYSTEM_NAME} LIBCLANG_OS_DIR)

# Install compiler-rt headers.
install(DIRECTORY include/
  DESTINATION ${LIBCLANG_INSTALL_PATH}/include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

# Call add_clang_compiler_rt_libraries to make sure that targets are built
# and installed in the directories where Clang driver expects to find them.
macro(add_clang_compiler_rt_libraries)
  # Setup output directories so that clang in build tree works.
  set_target_properties(${ARGN} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY
      ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION}/lib/${LIBCLANG_OS_DIR}
    LIBRARY_OUTPUT_DIRECTORY
      ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION}/lib/${LIBCLANG_OS_DIR}
    )
  # Add installation command.
  install(TARGETS ${ARGN}
    ARCHIVE DESTINATION ${LIBCLANG_INSTALL_PATH}/lib/${LIBCLANG_OS_DIR}
    LIBRARY DESTINATION ${LIBCLANG_INSTALL_PATH}/lib/${LIBCLANG_OS_DIR}
    )
endmacro(add_clang_compiler_rt_libraries)

# Add the public header's directory to the includes for all of compiler-rt.
include_directories(include)

add_subdirectory(lib)

if(LLVM_INCLUDE_TESTS)
  # Currently the tests have not been ported to CMake, so disable this
  # directory.
  #
  #add_subdirectory(test)
endif()
