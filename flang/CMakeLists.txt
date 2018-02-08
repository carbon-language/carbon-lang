cmake_minimum_required(VERSION 3.4.3)

# If we are not building as a part of LLVM, build FLANG as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(FLANG)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  find_program(LLVM_CONFIG "llvm-config")
  if(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--assertion-mode"
      "--bindir"
      "--libdir"
      "--includedir"
      "--prefix"
      "--src-root"
      "--cmakedir")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()
  else()
    message(FATAL_ERROR "llvm-config not found -- ${LLVM_CONFIG}")
  endif()

  list(GET CONFIG_OUTPUT 0 ENABLE_ASSERTIONS)
  list(GET CONFIG_OUTPUT 1 TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 INCLUDE_DIR)
  list(GET CONFIG_OUTPUT 4 LLVM_OBJ_ROOT)
  list(GET CONFIG_OUTPUT 5 MAIN_SRC_DIR)
  list(GET CONFIG_OUTPUT 6 LLVM_CONFIG_CMAKE_PATH)

  if(NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  # Normalize LLVM_CMAKE_PATH. --cmakedir might contain backslashes.
  # CMake assumes slashes as PATH.
  file(TO_CMAKE_PATH ${LLVM_CONFIG_CMAKE_PATH} LLVM_CMAKE_PATH)

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(LLVMCONFIG_FILE "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
  endif()

  # They are used as destination of target generators.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
  if(WIN32 OR CYGWIN)
    # DLL platform -- put DLLs into bin.
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  else()
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  endif()

  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)
  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)

  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)
  option(FLANG_ENABLE_BOOTSTRAP "Generate the flang bootstrap target" OFF)

  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
  include(VersionFromVCS)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ON)
  endif()

  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${LLVM_LIBRARY_DIR}")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX} )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX} )

  if(LLVM_INCLUDE_TESTS)
    set(Python_ADDITIONAL_VERSIONS 2.7)
    include(FindPythonInterp)
    if(NOT PYTHONINTERP_FOUND)
      message(FATAL_ERROR
"Unable to find Python interpreter, required for builds and testing.

Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
    endif()

    if( ${PYTHON_VERSION_STRING} VERSION_LESS 2.7 )
      message(FATAL_ERROR "Python 2.7 or newer is required")
    endif()

    # Check prebuilt llvm/utils.
    if(EXISTS ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/count${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/not${CMAKE_EXECUTABLE_SUFFIX})
      set(LLVM_UTILS_PROVIDED ON)
    endif()

    if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      # Note: path not really used, except for checking if lit was found
      set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      if(NOT LLVM_UTILS_PROVIDED)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/FileCheck utils/FileCheck)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/count utils/count)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/not utils/not)
        set(LLVM_UTILS_PROVIDED ON)
        set(FLANG_TEST_DEPS FileCheck count not)
      endif()
      set(UNITTEST_DIR ${LLVM_MAIN_SRC_DIR}/utils/unittest)
      if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h
          AND NOT EXISTS ${LLVM_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
          AND EXISTS ${UNITTEST_DIR}/CMakeLists.txt)
        add_subdirectory(${UNITTEST_DIR} utils/unittest)
      endif()
    else()
      # Seek installed Lit.
      find_program(LLVM_LIT
                   NAMES llvm-lit lit.py lit
                   PATHS "${LLVM_MAIN_SRC_DIR}/utils/lit"
                   DOC "Path to lit.py")
    endif()

    if(LLVM_LIT)
      # Define the default arguments to use with 'lit', and an option for the user
      # to override.
      set(LIT_ARGS_DEFAULT "-sv")
      if (MSVC OR XCODE)
        set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
      endif()
      set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

      # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
      if( WIN32 AND NOT CYGWIN )
        set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
      endif()
    else()
      set(LLVM_INCLUDE_TESTS OFF)
    endif()
  endif()

  set( FLANG_BUILT_STANDALONE 1 )
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
endif()

# Make sure that our source directory is on the current cmake module path so that
# we can include cmake files from this directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(LibXml2 2.5.3 QUIET)
if (LIBXML2_FOUND)
  set(FLANG_HAVE_LIBXML 1)
endif()

#!! find_package(Z3 4.5)

include(CheckIncludeFile)
check_include_file(sys/resource.h FLANG_HAVE_RLIMITS)

set(FLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the FLANG binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories flang will search for headers.")

set(GCC_INSTALL_PREFIX "" CACHE PATH "Directory where gcc is installed." )
set(DEFAULT_SYSROOT "" CACHE PATH
  "Default <path> to all compiler invocations for --sysroot=<path>." )

set(ENABLE_LINKER_BUILD_ID OFF CACHE BOOL "pass --build-id to ld")

set(ENABLE_X86_RELAX_RELOCATIONS OFF CACHE BOOL
    "enable x86 relax relocations by default")

set(FLANG_DEFAULT_LINKER "" CACHE STRING
  "Default linker to use (linker name or absolute path, empty for platform default)")

set(FLANG_DEFAULT_CXX_STDLIB "" CACHE STRING
  "Default C++ stdlib to use (\"libstdc++\" or \"libc++\", empty for platform default")
if (NOT(FLANG_DEFAULT_CXX_STDLIB STREQUAL "" OR
        FLANG_DEFAULT_CXX_STDLIB STREQUAL "libstdc++" OR
        FLANG_DEFAULT_CXX_STDLIB STREQUAL "libc++"))
  message(WARNING "Resetting default C++ stdlib to use platform default")
  set(FLANG_DEFAULT_CXX_STDLIB "" CACHE STRING
    "Default C++ stdlib to use (\"libstdc++\" or \"libc++\", empty for platform default" FORCE)
endif()

set(FLANG_DEFAULT_RTLIB "" CACHE STRING
  "Default runtime library to use (\"libgcc\" or \"compiler-rt\", empty for platform default)")
if (NOT(FLANG_DEFAULT_RTLIB STREQUAL "" OR
        FLANG_DEFAULT_RTLIB STREQUAL "libgcc" OR
        FLANG_DEFAULT_RTLIB STREQUAL "compiler-rt"))
  message(WARNING "Resetting default rtlib to use platform default")
  set(FLANG_DEFAULT_RTLIB "" CACHE STRING
    "Default runtime library to use (\"libgcc\" or \"compiler-rt\", empty for platform default)" FORCE)
endif()

set(FLANG_DEFAULT_OPENMP_RUNTIME "libomp" CACHE STRING
  "Default OpenMP runtime used by -fopenmp.")

set(FLANG_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific text for showing with version information.")

if( FLANG_VENDOR )
  add_definitions( -DFLANG_VENDOR="${FLANG_VENDOR} " )
endif()

set(FLANG_REPOSITORY_STRING "" CACHE STRING
  "Vendor-specific text for showing the repository the source is taken from.")

if(FLANG_REPOSITORY_STRING)
  add_definitions(-DFLANG_REPOSITORY_STRING="${FLANG_REPOSITORY_STRING}")
endif()

set(FLANG_VENDOR_UTI "org.llvm.flang" CACHE STRING
  "Vendor-specific uti.")

# The libdir suffix must exactly match whatever LLVM's configuration used.
set(FLANG_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
  file(GLOB_RECURSE
    tablegenned_files_on_include_dir
    "${FLANG_SOURCE_DIR}/include/flang/*.inc")
  if( tablegenned_files_on_include_dir )
    message(FATAL_ERROR "Apparently there is a previous in-source build, "
"probably as the result of running `configure' and `make' on "
"${FLANG_SOURCE_DIR}. This may cause problems. The suspicious files are:\n"
"${tablegenned_files_on_include_dir}\nPlease clean the source directory.")
  endif()
endif()

# If FLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED FLANG_VERSION_MAJOR)
  set(FLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED FLANG_VERSION_MINOR)
  set(FLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED FLANG_VERSION_PATCHLEVEL)
  set(FLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
# Unlike PACKAGE_VERSION, FLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "FLANG version: ${FLANG_VERSION}")

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/flang/Basic/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/flang/Basic/Version.inc)

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual")
  if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "FLANG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
  endif ()

  # Enable -pedantic for FLANG even if it's not enabled for LLVM.
  if (NOT LLVM_ENABLE_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wno-long-long")
  endif ()

  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if( CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types" )
  endif()
endif ()

# Determine HOST_LINK_VERSION on Darwin.
set(HOST_LINK_VERSION)
if (APPLE)
  set(LD_V_OUTPUT)
  execute_process(
    COMMAND sh -c "${CMAKE_LINKER} -v 2>&1 | head -1"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE LD_V_OUTPUT
  )
  if (NOT HAD_ERROR)
    if ("${LD_V_OUTPUT}" MATCHES ".*ld64-([0-9.]+).*")
      string(REGEX REPLACE ".*ld64-([0-9.]+).*" "\\1" HOST_LINK_VERSION ${LD_V_OUTPUT})
    elseif ("${LD_V_OUTPUT}" MATCHES "[^0-9]*([0-9.]+).*")
      string(REGEX REPLACE "[^0-9]*([0-9.]+).*" "\\1" HOST_LINK_VERSION ${LD_V_OUTPUT})
    endif()
  else()
    message(FATAL_ERROR "${CMAKE_LINKER} failed with status ${HAD_ERROR}")
  endif()
endif()

include(CMakeParseArguments)
include(AddFlang)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

#!! if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
#!!
#!!  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/flang
#!!    DESTINATION include
#!!    FILES_MATCHING
#!!    PATTERN "CMakeFiles" EXCLUDE
#!!    PATTERN "*.inc"
#!!    PATTERN "*.h"
#!!    )
#!!
#!!  install(PROGRAMS utils/bash-autocomplete.sh
#!!    DESTINATION share/flang
#!!    )
#!! endif()

add_definitions( -D_GNU_SOURCE )

option(FLANG_BUILD_TOOLS
  "Build the FLANG tools. If OFF, just generate build targets." ON)

option(FLANG_ENABLE_ARCMT "Build ARCMT." ON)
#!!option(FLANG_ENABLE_STATIC_ANALYZER "Build static analyzer." ON)
#!!
#!!option(FLANG_ANALYZER_BUILD_Z3
#!!  "Build the static analyzer with the Z3 constraint manager." OFF)
#!!
#!!if(NOT FLANG_ENABLE_STATIC_ANALYZER AND (FLANG_ENABLE_ARCMT OR FLANG_ANALYZER_BUILD_Z3))
#!!  message(FATAL_ERROR "Cannot disable static analyzer while enabling ARCMT or Z3")
#!!endif()
#!!
#!!if(FLANG_ANALYZER_BUILD_Z3)
#!!  if(Z3_FOUND)
#!!    set(FLANG_ANALYZER_WITH_Z3 1)
#!!  else()
#!!    message(FATAL_ERROR "Cannot find Z3 header file or shared library")
#!!  endif()
#!!endif()
#!!
#!!if(FLANG_ENABLE_ARCMT)
#!!  set(FLANG_ENABLE_OBJC_REWRITER ON)
#!!endif()

# FLANG version information
set(FLANG_EXECUTABLE_VERSION
     "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the flang executable, in the form XX.YY")
set(LIBFLANG_LIBRARY_VERSION
     "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the libflang library , in the form XX.YY")
mark_as_advanced(FLANG_EXECUTABLE_VERSION LIBFLANG_LIBRARY_VERSION)

option(FLANG_INCLUDE_TESTS
       "Generate build targets for the FLANG unit tests."
       ${LLVM_INCLUDE_TESTS})

add_subdirectory(utils/TableGen)

add_subdirectory(include)

# All targets below may depend on all tablegen'd files.
get_property(FLANG_TABLEGEN_TARGETS GLOBAL PROPERTY FLANG_TABLEGEN_TARGETS)
list(APPEND LLVM_COMMON_DEPENDS ${FLANG_TABLEGEN_TARGETS})

add_subdirectory(lib)
add_subdirectory(tools)
#!! add_subdirectory(runtime)

#!! option(FLANG_BUILD_EXAMPLES "Build FLANG example programs by default." OFF)
#!! add_subdirectory(examples)

if(APPLE)
  # this line is needed as a cleanup to ensure that any CMakeCaches with the old
  # default value get updated to the new default.
  if(FLANG_ORDER_FILE STREQUAL "")
    unset(FLANG_ORDER_FILE CACHE)
    unset(FLANG_ORDER_FILE)
  endif()


  set(FLANG_ORDER_FILE ${CMAKE_CURRENT_BINARY_DIR}/flang.order CACHE FILEPATH
    "Order file to use when compiling flang in order to improve startup time (Darwin Only - requires ld64).")

  if(NOT EXISTS ${FLANG_ORDER_FILE})
    string(FIND "${FLANG_ORDER_FILE}" "${CMAKE_CURRENT_BINARY_DIR}" PATH_START)
    if(PATH_START EQUAL 0)
      file(WRITE ${FLANG_ORDER_FILE} "\n")
    else()
      message(FATAL_ERROR "Specified order file '${FLANG_ORDER_FILE}' does not exist.")
    endif()
  endif()
endif()


#!! if( FLANG_INCLUDE_TESTS )
#!!   if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
#!!     add_subdirectory(unittests)
#!!     list(APPEND FLANG_TEST_DEPS FlangUnitTests)
#!!     list(APPEND FLANG_TEST_PARAMS
#!!       flang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/Unit/lit.site.cfg
#!!       )
#!!   endif()
#!!   add_subdirectory(test)
#!! 
#!!   if(FLANG_BUILT_STANDALONE)
#!!     # Add a global check rule now that all subdirectories have been traversed
#!!     # and we know the total set of lit testsuites.
#!!     get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
#!!     get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
#!!     get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
#!!     get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
#!!     add_lit_target(check-all
#!!       "Running all regression tests"
#!!       ${LLVM_LIT_TESTSUITES}
#!!       PARAMS ${LLVM_LIT_PARAMS}
#!!       DEPENDS ${LLVM_LIT_DEPENDS}
#!!       ARGS ${LLVM_LIT_EXTRA_ARGS}
#!!       )
#!!   endif()
#!!   add_subdirectory(utils/perf-training)
#!! endif()

#!! option(FLANG_INCLUDE_DOCS "Generate build targets for the FLANG docs."
#!!   ${LLVM_INCLUDE_DOCS})
#!! if( FLANG_INCLUDE_DOCS )
#!!   add_subdirectory(docs)
#!! endif()

add_subdirectory(cmake/modules)

if(FLANG_STAGE)
  message(STATUS "Setting current flang stage to: ${FLANG_STAGE}")
endif()


#if (LLVM_ADD_NATIVE_VISUALIZERS_TO_SOLUTION)
#  add_subdirectory(utils/FlangVisualizers)
#endif()

configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h)