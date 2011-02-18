# If we are not building as a part of LLVM, build Clang as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(Clang)
  cmake_minimum_required(VERSION 2.8)

  set(CLANG_PATH_TO_LLVM_SOURCE "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
  set(CLANG_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")

  if( CLANG_PATH_TO_LLVM_SOURCE )
    if( NOT EXISTS "${CLANG_PATH_TO_LLVM_SOURCE}/cmake/config-ix.cmake" )
      message(FATAL_ERROR "Please set CLANG_PATH_TO_LLVM_SOURCE to the root directory of LLVM source code.")
    else()
      get_filename_component(LLVM_MAIN_SRC_DIR ${CLANG_PATH_TO_LLVM_SOURCE}
	ABSOLUTE)
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()

  if( NOT EXISTS "${CLANG_PATH_TO_LLVM_BUILD}/bin/tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
    message(FATAL_ERROR "Please set CLANG_PATH_TO_LLVM_BUILD to a directory containing a LLVM build.")
  endif()

  list(APPEND CMAKE_MODULE_PATH "${CLANG_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${CLANG_PATH_TO_LLVM_BUILD}
    ABSOLUTE)

  include(AddLLVM)
  include(TableGen)
  include("${CLANG_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVM.cmake")
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${PATH_TO_LLVM_BUILD}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  if( NOT PATH_TO_LLVM_BUILD STREQUAL LLVM_MAIN_SRC_DIR )
    include_directories("${LLVM_MAIN_INCLUDE_DIR}")
  endif()
  link_directories("${PATH_TO_LLVM_BUILD}/lib")

  set(LLVM_TABLEGEN_EXE "${PATH_TO_LLVM_BUILD}/bin/tblgen")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

  add_definitions( -D__STDC_LIMIT_MACROS )
  add_definitions( -D__STDC_CONSTANT_MACROS )

  set( CLANG_BUILT_STANDALONE 1 )
endif()

set(CLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the Clang binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clang will search for headers.")

set(CLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

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
    "${CLANG_SOURCE_DIR}/include/clang/*.inc")
  if( tablegenned_files_on_include_dir )
    message(FATAL_ERROR "Apparently there is a previous in-source build, "
"probably as the result of running `configure' and `make' on "
"${CLANG_SOURCE_DIR}. This may cause problems. The suspicious files are:\n"
"${tablegenned_files_on_include_dir}\nPlease clean the source directory.")
  endif()
endif()

# Compute the Clang version from the LLVM version.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
  ${PACKAGE_VERSION})
message(STATUS "Clang version: ${CLANG_VERSION}")

string(REGEX REPLACE "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" "\\1" CLANG_VERSION_MAJOR
  ${CLANG_VERSION})
string(REGEX REPLACE "[0-9]+\\.([0-9]+)(\\.[0-9]+)?" "\\1" CLANG_VERSION_MINOR
  ${CLANG_VERSION})
if (${CLANG_VERSION} MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
  set(CLANG_HAS_VERSION_PATCHLEVEL 1)
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" CLANG_VERSION_PATCHLEVEL
    ${CLANG_VERSION})
else()
  set(CLANG_HAS_VERSION_PATCHLEVEL 0)
endif()

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/clang/Basic/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/clang/Basic/Version.inc)

# Add appropriate flags for GCC
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
endif ()

if (APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif ()

configure_file(
  ${CLANG_SOURCE_DIR}/include/clang/Config/config.h.cmake
  ${CLANG_BINARY_DIR}/include/clang/Config/config.h)

macro(add_clang_library name)
  llvm_process_sources(srcs ${ARGN})
  if(MSVC_IDE OR XCODE)
    string( REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list( GET split_path -1 dir)
    file( GLOB_RECURSE headers
      ../../../include/clang/StaticAnalyzer${dir}/*.h
      ../../../include/clang/StaticAnalyzer${dir}/*.td
      ../../../include/clang/StaticAnalyzer${dir}/*.def
      ../../include/clang${dir}/*.h
      ../../include/clang${dir}/*.td
      ../../include/clang${dir}/*.def)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE OR XCODE)
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind)
  endif()
  add_library( ${name} ${libkind} ${srcs} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  if( LLVM_USED_LIBS )
    foreach(lib ${LLVM_USED_LIBS})
      target_link_libraries( ${name} ${lib} )
    endforeach(lib)
  endif( LLVM_USED_LIBS )
  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  if (LLVM_COMMON_LIBS)
    target_link_libraries(${name} ${LLVM_COMMON_LIBS})
  endif()
  if( NOT MINGW )
    get_system_libs(llvm_system_libs)
    if( llvm_system_libs )
      target_link_libraries(${name} ${llvm_system_libs})
    endif()
  endif()
  add_dependencies(${name} ClangDiagnosticCommon)
  if(MSVC)
    get_target_property(cflag ${name} COMPILE_FLAGS)
    if(NOT cflag)
      set(cflag "")
    endif(NOT cflag)
    set(cflag "${cflag} /Za")
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS ${cflag})
  endif(MSVC)
  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
endmacro(add_clang_library)

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
endmacro(add_clang_executable)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.td"
  PATTERN ".svn" EXCLUDE
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "*.inc"
  )

add_definitions( -D_GNU_SOURCE -DHAVE_CLANG_CONFIG_H )

option(CLANG_BUILD_EXAMPLES "Build CLANG example programs." OFF)
if(CLANG_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)

# TODO: docs.
if( LLVM_INCLUDE_TESTS )
add_subdirectory(test)
endif()

# FIXME: unittests require gtest.
if( NOT CLANG_BUILT_STANDALONE )
  add_subdirectory(unittests)
endif()

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLANG_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLANG_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/Clang.sln")
  if( EXISTS "${CLANG_SLN_FILENAME}" )
    file(APPEND "${CLANG_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()
