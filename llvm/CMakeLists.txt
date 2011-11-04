# See docs/CMake.html for instructions about how to build LLVM with CMake.

project(LLVM)
cmake_minimum_required(VERSION 2.8)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

set(PACKAGE_VERSION "3.0")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(VersionFromVCS)

option(LLVM_APPEND_VC_REV
  "Append the version control system revision id to LLVM version" OFF)

if( LLVM_APPEND_VC_REV )
  add_version_info_from_vcs(PACKAGE_VERSION)
endif()

set(PACKAGE_NAME llvm)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed.
CMake would overwrite the makefiles distributed with LLVM.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

set(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVM_MAIN_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)
set(LLVM_EXAMPLES_BINARY_DIR ${LLVM_BINARY_DIR}/examples)
set(LLVM_LIBDIR_SUFFIX "" CACHE STRING "Define suffix of library directory name (32/64)" )

if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
  file(GLOB_RECURSE
    tablegenned_files_on_include_dir
    "${LLVM_MAIN_SRC_DIR}/include/llvm/*.gen")
  file(GLOB_RECURSE
    tablegenned_files_on_lib_dir
    "${LLVM_MAIN_SRC_DIR}/lib/Target/*.inc")
  if( tablegenned_files_on_include_dir OR tablegenned_files_on_lib_dir)
    message(FATAL_ERROR "Apparently there is a previous in-source build,
probably as the result of running `configure' and `make' on
${LLVM_MAIN_SRC_DIR}.
This may cause problems. The suspicious files are:
${tablegenned_files_on_lib_dir}
${tablegenned_files_on_include_dir}
Please clean the source directory.")
  endif()
endif()

set(LLVM_ALL_TARGETS
  ARM
  CBackend
  CellSPU
  CppBackend
  Mips
  MBlaze
  MSP430
  PowerPC
  PTX
  Sparc
  X86
  XCore
  )

# List of targets with JIT support:
set(LLVM_TARGETS_WITH_JIT X86 PowerPC ARM Mips)

if( MSVC )
  set(LLVM_TARGETS_TO_BUILD X86
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
else( MSVC )
  set(LLVM_TARGETS_TO_BUILD ${LLVM_ALL_TARGETS}
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
endif( MSVC )

option(LLVM_ENABLE_CBE_PRINTF_A "Set to ON if CBE is enabled for printf %a output" ON)
if(LLVM_ENABLE_CBE_PRINTF_A)
  set(ENABLE_CBE_PRINTF_A 1)
endif()

option(LLVM_ENABLE_TIMESTAMPS "Enable embedding timestamp information in build" ON)
if(LLVM_ENABLE_TIMESTAMPS)
  set(ENABLE_TIMESTAMPS 1)
endif()

option(LLVM_ENABLE_FFI "Use libffi to call external functions from the interpreter" OFF)
set(FFI_LIBRARY_DIR "" CACHE PATH "Additional directory, where CMake should search for libffi.so")
set(FFI_INCLUDE_DIR "" CACHE PATH "Additional directory, where CMake should search for ffi.h or ffi/ffi.h")

set(LLVM_TARGET_ARCH "host"
  CACHE STRING "Set target to use for LLVM JIT or use \"host\" for automatic detection.")

option(LLVM_ENABLE_THREADS "Use threads if available." ON)

if( LLVM_TARGETS_TO_BUILD STREQUAL "all" )
  set( LLVM_TARGETS_TO_BUILD ${LLVM_ALL_TARGETS} )
endif()

set(LLVM_ENUM_TARGETS "")
foreach(c ${LLVM_TARGETS_TO_BUILD})
  list(FIND LLVM_ALL_TARGETS ${c} idx)
  if( idx LESS 0 )
    message(FATAL_ERROR "The target `${c}' does not exist.
    It should be one of\n${LLVM_ALL_TARGETS}")
  else()
    set(LLVM_ENUM_TARGETS "${LLVM_ENUM_TARGETS}LLVM_TARGET(${c})\n")
  endif()
endforeach(c)

# Produce llvm/Config/Targets.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/Targets.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/Targets.def
  )

set(llvm_builded_incs_dir ${LLVM_BINARY_DIR}/include/llvm)

include(AddLLVMDefinitions)

option(LLVM_ENABLE_PIC "Build Position-Independent Code" ON)

# MSVC has a gazillion warnings with this.
if( MSVC )
  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." OFF)
else( MSVC )
  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)
endif()

option(LLVM_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LLVM_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)

if( uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
  option(LLVM_ENABLE_ASSERTIONS "Enable assertions" OFF)
else()
  option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
endif()

# All options referred to from HandleLLVMOptions have to be specified
# BEFORE this include, otherwise options will not be correctly set on
# first cmake run
include(config-ix)
include(HandleLLVMOptions)

configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/config.h.cmake
  ${LLVM_BINARY_DIR}/include/llvm/Config/config.h)

configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/llvm-config.h.cmake
  ${LLVM_BINARY_DIR}/include/llvm/Config/llvm-config.h)

configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support/DataTypes.h.cmake
  ${LLVM_BINARY_DIR}/include/llvm/Support/DataTypes.h)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories( ${LLVM_BINARY_DIR}/include ${LLVM_MAIN_INCLUDE_DIR})

if( ${CMAKE_SYSTEM_NAME} MATCHES SunOS )
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include llvm/Support/Solaris.h")
endif( ${CMAKE_SYSTEM_NAME} MATCHES SunOS )

include(AddLLVM)
include(TableGen)

if( MINGW )
  # People report that -O3 is unreliable on MinGW. The traditional
  # build also uses -O2 for that reason:
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O2")
endif()

# Put this before tblgen. Else we have a circular dependence.
add_subdirectory(lib/Support)
add_subdirectory(lib/TableGen)

add_subdirectory(utils/TableGen)

add_subdirectory(include/llvm)

add_subdirectory(lib)

add_subdirectory(utils/FileCheck)
add_subdirectory(utils/FileUpdate)
add_subdirectory(utils/count)
add_subdirectory(utils/not)
add_subdirectory(utils/llvm-lit)

add_subdirectory(projects)

option(LLVM_BUILD_TOOLS
  "Build the LLVM tools. If OFF, just generate build targets." ON)
option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
if( LLVM_INCLUDE_TOOLS )
  add_subdirectory(tools)
endif()

option(LLVM_BUILD_RUNTIME
  "Build the LLVM runtime libraries. If OFF, just generate build targets." ON)
option(LLVM_INCLUDE_RUNTIME "Generate build targets for the LLVM runtimes" ON)
if( LLVM_INCLUDE_RUNTIME )
  add_subdirectory(runtime)
endif()

option(LLVM_BUILD_EXAMPLES
  "Build the LLVM example programs. If OFF, just generate build targets." OFF)
option(LLVM_INCLUDE_EXAMPLES "Generate build targets for the LLVM examples" ON)
if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()

option(LLVM_BUILD_TESTS
  "Build LLVM unit tests. If OFF, just generate build targets." OFF)
if( LLVM_INCLUDE_TESTS )
  add_subdirectory(test)
  add_subdirectory(utils/unittest)
  add_subdirectory(unittests)
  if (MSVC)
    # This utility is used to prevent chrashing tests from calling Dr. Watson on
    # Windows.
    add_subdirectory(utils/KillTheDoctor)
  endif()
endif()

add_subdirectory(cmake/modules)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.td"
  PATTERN "*.inc"
  PATTERN "LICENSE.TXT"
  PATTERN ".svn" EXCLUDE
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  # Exclude include/llvm/CMakeFiles/intrinsics_gen.dir, matched by "*.def"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN ".svn" EXCLUDE
  )

# TODO: make and install documentation.

set(CPACK_PACKAGE_VENDOR "LLVM")
set(CPACK_PACKAGE_VERSION_MAJOR 2)
set(CPACK_PACKAGE_VERSION_MINOR 9)
add_version_info_from_vcs(CPACK_PACKAGE_VERSION_PATCH)
include(CPack)

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if(MSVC_VERSION EQUAL 1600)
  set(LLVM_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/LLVM.sln")
  if( EXISTS "${LLVM_SLN_FILENAME}" )
    file(APPEND "${LLVM_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()
