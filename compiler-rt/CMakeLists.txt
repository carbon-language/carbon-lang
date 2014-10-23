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
  project(CompilerRT C CXX)
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
  # Use just-built Clang to compile/link tests on all platforms, except for
  # Windows where we need to use clang-cl instead.
  if(NOT MSVC)
    set(COMPILER_RT_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
  else()
    set(COMPILER_RT_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang.exe)
  endif()
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
  set(COMPILER_RT_TEST_COMPILER ${CMAKE_C_COMPILER} CACHE PATH "Compiler to use for testing")

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
  file(TO_CMAKE_PATH ${LLVM_BINARY_DIR} LLVM_BINARY_DIR_CMAKE_STYLE)
  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR_CMAKE_STYLE}/share/llvm/cmake")
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

if("${COMPILER_RT_TEST_COMPILER}" MATCHES "clang[+]*$")
  set(COMPILER_RT_TEST_COMPILER_ID Clang)
elseif("${COMPILER_RT_TEST_COMPILER}" MATCHES "clang.*.exe$")
  set(COMPILER_RT_TEST_COMPILER_ID Clang)
else()
  set(COMPILER_RT_TEST_COMPILER_ID GNU)
endif()

# Tests using XFAIL use the first value in COMPILER_RT_TEST_TARGET_TRIPLE
set(COMPILER_RT_TEST_TARGET_TRIPLE ${TARGET_TRIPLE} CACHE STRING
    "Default triple for cross-compiled executables")
string(REPLACE "-" ";" TARGET_TRIPLE_LIST ${COMPILER_RT_TEST_TARGET_TRIPLE})
list(GET TARGET_TRIPLE_LIST 0 COMPILER_RT_TEST_TARGET_ARCH)
list(GET TARGET_TRIPLE_LIST 1 COMPILER_RT_TEST_TARGET_OS)
list(GET TARGET_TRIPLE_LIST 2 COMPILER_RT_TEST_TARGET_ABI)

if ("${COMPILER_RT_TEST_TARGET_ABI}" STREQUAL "androideabi")
  set(ANDROID 1)
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

set(COMPILER_RT_EXTRA_ANDROID_HEADERS ${COMPILER_RT_SOURCE_DIR}/android/include)

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

function(get_target_flags_for_arch arch out_var)
  list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
  if(ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "Unsupported architecture: ${arch}")
  else()
    set(${out_var} ${TARGET_${arch}_CFLAGS} PARENT_SCOPE)
  endif()
endfunction()

# We support running instrumented tests when we're not cross compiling
# and target a UNIX-like system or Windows.
# We can run tests on Android even when we are cross-compiling.
if(("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND (UNIX OR MSVC)) OR ANDROID
   OR COMPILER_RT_EMULATOR)
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" ON)
else()
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" OFF)
endif()

option(COMPILER_RT_DEBUG "Build runtimes with full debug info" OFF)
# COMPILER_RT_DEBUG_PYBOOL is used by lit.common.configured.in.
pythonize_bool(COMPILER_RT_DEBUG)

# We have to support both static and dynamic/shared runtime on Windows.
# Android only works with dynamic runtime.
if(WIN32 OR ANDROID)
option(COMPILER_RT_BUILD_SHARED_ASAN "Build shared version of AddressSanitizer runtime" ON)
else()
option(COMPILER_RT_BUILD_SHARED_ASAN "Build shared version of AddressSanitizer runtime" OFF)
endif()

#================================
# Setup Compiler Flags
#================================
include(config-ix)

if(MSVC)
  append_string_if(COMPILER_RT_HAS_W3_FLAG /W3 CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
else()
  append_string_if(COMPILER_RT_HAS_WALL_FLAG -Wall CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()
if(COMPILER_RT_ENABLE_WERROR)
  append_string_if(COMPILER_RT_HAS_WERROR_FLAG -Werror CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append_string_if(COMPILER_RT_HAS_WX_FLAG /WX CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

append_string_if(COMPILER_RT_HAS_STD_CXX11_FLAG -std=c++11 CMAKE_CXX_FLAGS)

# Emulate C99 and C++11's __func__ for MSVC prior to 2013 CTP.
if(NOT COMPILER_RT_HAS_FUNC_SYMBOL)
  add_definitions(-D__func__=__FUNCTION__)
endif()

# Provide some common commmandline flags for Sanitizer runtimes.
append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_BUILTIN_FLAG -fno-builtin SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG -fomit-frame-pointer SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FUNWIND_TABLES_FLAG -funwind-tables SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG -fno-stack-protector SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG -fno-function-sections SANITIZER_COMMON_CFLAGS)

if(MSVC)
  # Replace the /MD[d] flags with /MT.
  # FIXME: In fact, sanitizers should support both /MT and /MD, see PR20214.
  if(COMPILER_RT_HAS_MT_FLAG)
    foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      elseif(${flag_var} MATCHES "/MDd")
        string(REGEX REPLACE "/MDd" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  endif()
  append_list_if(COMPILER_RT_HAS_Oy_FLAG /Oy- SANITIZER_COMMON_CFLAGS)
  append_list_if(COMPILER_RT_HAS_GS_FLAG /GS- SANITIZER_COMMON_CFLAGS)
endif()

# Build with optimization, unless we're in debug mode. If we're using MSVC,
# always respect the optimization flags set by CMAKE_BUILD_TYPE instead.
if(NOT COMPILER_RT_DEBUG AND NOT MSVC)
  list(APPEND SANITIZER_COMMON_CFLAGS -O3)
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
append_list_if(COMPILER_RT_HAS_WNO_GNU_FLAG -Wno-gnu SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WNO_VARIADIC_MACROS_FLAG -Wno-variadic-macros SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WNO_C99_EXTENSIONS_FLAG -Wno-c99-extensions SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WNO_NON_VIRTUAL_DTOR_FLAG -Wno-non-virtual-dtor SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4146_FLAG /wd4146 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4291_FLAG /wd4291 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4391_FLAG /wd4391 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4722_FLAG /wd4722 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4800_FLAG /wd4800 SANITIZER_COMMON_CFLAGS)
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

  set(SANITIZER_MIN_OSX_VERSION 10.7)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "") # We're setting the flag manually below.
  set(DARWIN_osx_CFLAGS -mmacosx-version-min=${SANITIZER_MIN_OSX_VERSION})
  set(DARWIN_iossim_CFLAGS
    -mios-simulator-version-min=7.0 -isysroot ${IOSSIM_SDK_DIR})
  set(DARWIN_osx_LINKFLAGS)
  set(DARWIN_iossim_LINKFLAGS
    -Wl,-ios_simulator_version_min,7.0.0
    -mios-simulator-version-min=7.0
    -isysroot ${IOSSIM_SDK_DIR})
endif()

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
