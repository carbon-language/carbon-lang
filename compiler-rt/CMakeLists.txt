# CMake build for CompilerRT.
#
# This build assumes that CompilerRT is checked out into the
# 'projects/compiler-rt' inside of an LLVM tree.
# Standalone build system for CompilerRT is not yet ready.
#
# An important constraint of the build is that it only produces libraries
# based on the ability of the host toolchain to target various platforms.

# Check if compiler-rt is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(CompilerRT)
  set(COMPILER_RT_STANDALONE_BUILD TRUE)
else()
  set(COMPILER_RT_STANDALONE_BUILD FALSE)
endif()

# The CompilerRT build system requires CMake version 2.8.8 or higher in order
# to use its support for building convenience "libraries" as a collection of
# .o files. This is particularly useful in producing larger, more complex
# runtime libraries.
if (NOT MSVC)
  cmake_minimum_required(VERSION 2.8.8)
else()
  # Version 2.8.12.1 is required to build with Visual Studion 2013.
  cmake_minimum_required(VERSION 2.8.12.1)
endif()

# FIXME: It may be removed when we use 2.8.12.
if(CMAKE_VERSION VERSION_LESS 2.8.12)
  # Invalidate a couple of keywords.
  set(cmake_2_8_12_INTERFACE)
  set(cmake_2_8_12_PRIVATE)
else()
  # Use ${cmake_2_8_12_KEYWORD} intead of KEYWORD in target_link_libraries().
  set(cmake_2_8_12_INTERFACE INTERFACE)
  set(cmake_2_8_12_PRIVATE PRIVATE)
  if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW) # automatic when 2.8.12 is required
  endif()
endif()

# Top level target used to build all compiler-rt libraries.
add_custom_target(compiler-rt ALL)

if (NOT COMPILER_RT_STANDALONE_BUILD)
  # Compute the Clang version from the LLVM version.
  # FIXME: We should be able to reuse CLANG_VERSION variable calculated
  #        in Clang cmake files, instead of copying the rules here.
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
         ${PACKAGE_VERSION})
  # Setup the paths where compiler-rt runtimes and headers should be stored.
  set(COMPILER_RT_OUTPUT_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION})
  set(COMPILER_RT_EXEC_OUTPUT_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  set(COMPILER_RT_INSTALL_PATH lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION})
  option(COMPILER_RT_INCLUDE_TESTS "Generate and build compiler-rt unit tests."
         ${LLVM_INCLUDE_TESTS})
 option(COMPILER_RT_ENABLE_WERROR "Fail and stop if warning is triggered"
        ${LLVM_ENABLE_WERROR})
  # Use just-built Clang to compile/link tests.
  set(COMPILER_RT_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
  set(COMPILER_RT_TEST_COMPILER_ID Clang)
else()
  # Take output dir and install path from the user.
  set(COMPILER_RT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH
    "Path where built compiler-rt libraries should be stored.")
  set(COMPILER_RT_EXEC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH
    "Path where built compiler-rt executables should be stored.")
  set(COMPILER_RT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX} CACHE PATH
    "Path where built compiler-rt libraries should be installed.")
  option(COMPILER_RT_INCLUDE_TESTS "Generate and build compiler-rt unit tests." OFF)
  option(COMPILER_RT_ENABLE_WERROR "Fail and stop if warning is triggered" OFF)
  # Use a host compiler to compile/link tests.
  set(COMPILER_RT_TEST_COMPILER ${CMAKE_C_COMPILER})
  set(COMPILER_RT_TEST_COMPILER_ID ${CMAKE_C_COMPILER_ID})

  if (NOT LLVM_CONFIG_PATH)
    find_program(LLVM_CONFIG_PATH "llvm-config"
                 DOC "Path to llvm-config binary")
    if (NOT LLVM_CONFIG_PATH)
      message(FATAL_ERROR "llvm-config not found: specify LLVM_CONFIG_PATH")
    endif()
  endif()
  execute_process(
    COMMAND ${LLVM_CONFIG_PATH} "--obj-root" "--bindir" "--libdir" "--src-root"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE CONFIG_OUTPUT)
  if (HAD_ERROR)
    message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
  endif()
  string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" CONFIG_OUTPUT ${CONFIG_OUTPUT})
  list(GET CONFIG_OUTPUT 0 LLVM_BINARY_DIR)
  list(GET CONFIG_OUTPUT 1 LLVM_TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 2 LLVM_LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 LLVM_MAIN_SRC_DIR)

  # Make use of LLVM CMake modules.
  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/share/llvm/cmake")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
  # Get some LLVM variables from LLVMConfig.
  include("${LLVM_CMAKE_PATH}/LLVMConfig.cmake")

  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib)

  # Find Python interpreter.
  set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5)
  include(FindPythonInterp)
  if(NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "
      Unable to find Python interpreter required testing. Please install Python
      or specify the PYTHON_EXECUTABLE CMake variable.")
  endif()

  # Define default arguments to lit.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
endif()

string(TOLOWER ${CMAKE_SYSTEM_NAME} COMPILER_RT_OS_DIR)
set(COMPILER_RT_LIBRARY_OUTPUT_DIR
  ${COMPILER_RT_OUTPUT_DIR}/lib/${COMPILER_RT_OS_DIR})
set(COMPILER_RT_LIBRARY_INSTALL_DIR
  ${COMPILER_RT_INSTALL_PATH}/lib/${COMPILER_RT_OS_DIR})

# Add path for custom compiler-rt modules.
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )
include(CompilerRTUtils)

set(COMPILER_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPILER_RT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
# Setup custom SDK sysroots.
set(COMPILER_RT_DARWIN_SDK_SYSROOT ${COMPILER_RT_SOURCE_DIR}/SDKs/darwin)
set(COMPILER_RT_LINUX_SDK_SYSROOT ${COMPILER_RT_SOURCE_DIR}/SDKs/linux)

set(COMPILER_RT_EXTRA_ANDROID_HEADERS ${COMPILER_RT_SOURCE_DIR}/third_party/android/include)

# Detect whether the current target platform is 32-bit or 64-bit, and setup
# the correct commandline flags needed to attempt to target 32-bit and 64-bit.
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 4 AND
    NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Please use architecture with 4 or 8 byte pointers.")
endif()
if (NOT MSVC)
  set(TARGET_64_BIT_CFLAGS "-m64")
  set(TARGET_32_BIT_CFLAGS "-m32")
else()
  set(TARGET_64_BIT_CFLAGS "")
  set(TARGET_32_BIT_CFLAGS "")
endif()

# List of architectures we can target.
set(COMPILER_RT_SUPPORTED_ARCH)

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
set(SIMPLE_SOURCE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple.c)
file(WRITE ${SIMPLE_SOURCE} "#include <stdlib.h>\nint main() {}")

# test_target_arch(<arch> <target flags...>)
# Sets the target flags for a given architecture and determines if this
# architecture is supported by trying to build a simple file.
macro(test_target_arch arch)
  set(TARGET_${arch}_CFLAGS ${ARGN})
  try_compile(CAN_TARGET_${arch} ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE}
              COMPILE_DEFINITIONS "${TARGET_${arch}_CFLAGS}"
              CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${TARGET_${arch}_CFLAGS}")
  if(${CAN_TARGET_${arch}})
    list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
  endif()
endmacro()

if(ANDROID_COMMON_FLAGS)
  test_target_arch(arm_android "${ANDROID_COMMON_FLAGS}")
else()
  if("${LLVM_NATIVE_ARCH}" STREQUAL "X86")
    if (NOT MSVC)
      test_target_arch(x86_64 ${TARGET_64_BIT_CFLAGS})
    endif()
    test_target_arch(i386 ${TARGET_32_BIT_CFLAGS})
  elseif("${LLVM_NATIVE_ARCH}" STREQUAL "PowerPC")
    test_target_arch(powerpc64 ${TARGET_64_BIT_CFLAGS})
  elseif("${LLVM_NATIVE_ARCH}" STREQUAL "ARM")
    test_target_arch(arm "")
  endif()
endif()

# We only support running instrumented tests when we're not cross compiling
# and target a unix-like system. We can run tests on Android even when we are
# cross-compiling.
if(("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX) OR ANDROID)
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" ON)
else()
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" OFF)
endif()

# Check if compiler-rt is built with libc++.
find_flag_in_string("${CMAKE_CXX_FLAGS}" "-stdlib=libc++"
                    COMPILER_RT_USES_LIBCXX)

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

option(COMPILER_RT_DEBUG "Build runtimes with full debug info" OFF)
# COMPILER_RT_DEBUG_PYBOOL is used by lit.common.configured.in.
pythonize_bool(COMPILER_RT_DEBUG)

#================================
# Setup Compiler Flags
#================================
include(config-ix)

if(COMPILER_RT_ENABLE_WERROR)
  add_definitions_if(COMPILER_RT_HAS_WERROR_FLAG -Werror)
  add_definitions_if(COMPILER_RT_HAS_WX_FLAG /WX)
endif()

# Emulate C99 and C++11's __func__ for MSVC prior to 2013 CTP.
if(NOT COMPILER_RT_HAS_FUNC_SYMBOL)
  add_definitions(-D__func__=__FUNCTION__)
endif()

# Provide some common commmandline flags for Sanitizer runtimes.
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FPIC_FLAG -fPIC)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FNO_BUILTIN_FLAG -fno-builtin)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG -fomit-frame-pointer)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FUNWIND_TABLES_FLAG -funwind-tables)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG -fno-stack-protector)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG -fno-function-sections)

if(MSVC)
  # Remove /MD flag so that it doesn't conflict with /MT.
  if(COMPILER_RT_HAS_MT_FLAG)
    string(REGEX REPLACE "(^| ) */MD *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    list(APPEND SANITIZER_COMMON_CFLAGS /MT)
  endif()
  append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_Oy_FLAG /Oy-)
  append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_GS_FLAG /GS-)
endif()

# Build with optimization, unless we're in debug mode.
if(NOT COMPILER_RT_DEBUG)
  if(MSVC)
    list(APPEND SANITIZER_COMMON_CFLAGS /O2)
  else()
    list(APPEND SANITIZER_COMMON_CFLAGS -O3)
  endif()
endif()

# Build sanitizer runtimes with debug info.
if(COMPILER_RT_HAS_GLINE_TABLES_ONLY_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -gline-tables-only)
elseif(COMPILER_RT_HAS_G_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -g)
elseif(COMPILER_RT_HAS_Zi_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS /Zi)
endif()

# Turn off several warnings.
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_WNO_GNU_FLAG -Wno-gnu)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_WNO_VARIADIC_MACROS_FLAG -Wno-variadic-macros)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_WNO_C99_EXTENSIONS_FLAG -Wno-c99-extensions)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_WNO_NON_VIRTUAL_DTOR_FLAG -Wno-non-virtual-dtor)
append_if(SANITIZER_COMMON_CFLAGS COMPILER_RT_HAS_WD4722_FLAG /wd4722)

if(APPLE)
  # Obtain the iOS Simulator SDK path from xcodebuild.
  execute_process(
    COMMAND xcodebuild -version -sdk iphonesimulator Path
    OUTPUT_VARIABLE IOSSIM_SDK_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "-mmacosx-version-min="
         MACOSX_VERSION_MIN_FLAG "${CMAKE_CXX_FLAGS}")
  set(SANITIZER_COMMON_SUPPORTED_DARWIN_OS osx)
  if (IOSSIM_SDK_DIR AND NOT MACOSX_VERSION_MIN_FLAG)
    list(APPEND SANITIZER_COMMON_SUPPORTED_DARWIN_OS iossim)
  endif()

  if(COMPILER_RT_USES_LIBCXX)
    set(SANITIZER_MIN_OSX_VERSION 10.7)
  else()
    set(SANITIZER_MIN_OSX_VERSION 10.6)
  endif()
  set(DARWIN_osx_CFLAGS -mmacosx-version-min=${SANITIZER_MIN_OSX_VERSION})
  set(DARWIN_iossim_CFLAGS 
    -mios-simulator-version-min=7.0 -isysroot ${IOSSIM_SDK_DIR})
  set(DARWIN_osx_LINKFLAGS)
  set(DARWIN_iossim_LINKFLAGS
    -Wl,-ios_simulator_version_min,7.0.0
    -mios-simulator-version-min=7.0
    -isysroot ${IOSSIM_SDK_DIR})
endif()

# Architectures supported by Sanitizer runtimes. Specific sanitizers may
# support only subset of these (e.g. TSan works on x86_64 only).
filter_available_targets(SANITIZER_COMMON_SUPPORTED_ARCH
  x86_64 i386 powerpc64 arm)
filter_available_targets(ASAN_SUPPORTED_ARCH x86_64 i386 powerpc64)
filter_available_targets(DFSAN_SUPPORTED_ARCH x86_64)
filter_available_targets(LSAN_SUPPORTED_ARCH x86_64)
filter_available_targets(MSAN_SUPPORTED_ARCH x86_64)
filter_available_targets(TSAN_SUPPORTED_ARCH x86_64)
filter_available_targets(UBSAN_SUPPORTED_ARCH x86_64 i386)

add_subdirectory(include)

set(COMPILER_RT_LIBCXX_PATH ${LLVM_MAIN_SRC_DIR}/projects/libcxx)
if(EXISTS ${COMPILER_RT_LIBCXX_PATH}/)
  set(COMPILER_RT_HAS_LIBCXX_SOURCES TRUE)
else()
  set(COMPILER_RT_HAS_LIBCXX_SOURCES FALSE)
endif()

add_subdirectory(lib)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(unittests)
endif()
add_subdirectory(test)
