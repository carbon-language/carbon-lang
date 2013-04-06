# If we are not building as a part of LLVM, build lld as a standalone project,
# using LLVM as an external library.



if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lld)
  cmake_minimum_required(VERSION 2.8)

  set(LLD_PATH_TO_LLVM_SOURCE "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
  set(LLD_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")

  if (LLD_PATH_TO_LLVM_SOURCE)
    if (NOT EXISTS "${LLD_PATH_TO_LLVM_SOURCE}/cmake/config-ix.cmake")
      message(FATAL_ERROR "Please set LLD_PATH_TO_LLVM_SOURCE to the root "
              "directory of LLVM source code.")
    else()
      get_filename_component(LLVM_MAIN_SRC_DIR ${LLD_PATH_TO_LLVM_SOURCE}
                             ABSOLUTE)
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLD_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${LLD_PATH_TO_LLVM_BUILD}
                         ABSOLUTE)

  include(AddLLVM)
  include(TableGen)
  include("${LLD_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVMConfig.cmake")
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${PATH_TO_LLVM_BUILD}/include"
                      "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${PATH_TO_LLVM_BUILD}/lib")

  if( EXISTS "${LLD_PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
    set(LLVM_TABLEGEN_EXE "${PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  else()
    # FIXME: This is an utter hack.
    set(LLVM_TABLEGEN_EXE "${PATH_TO_LLVM_BUILD}/bin/Debug/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  endif()

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  set(LLD_BUILT_STANDALONE 1)
endif()

set(LLD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLD_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

list (APPEND CMAKE_MODULE_PATH "${LLD_SOURCE_DIR}/cmake/modules")

option(LLD_USE_VTUNE
       "Enable VTune user task tracking."
       OFF)
if (LLD_USE_VTUNE)
  find_package(VTune)
  if (VTUNE_FOUND)
    include_directories(${VTune_INCLUDE_DIRS})
    list(APPEND LLVM_COMMON_LIBS ${VTune_LIBRARIES})
    add_definitions(-DLLD_HAS_VTUNE)
  endif()
endif()

# lld requires c++11 to build. Make sure that we have a compiler and standard
# library combination that can do that.
if (MSVC11)
  # Do nothing, we're good.
elseif (NOT MSVC)
  # gcc and clang require the -std=c++0x or -std=c++11 flag.
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
      "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (NOT ("${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+0x" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=gnu\\+\\+0x" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+11" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=gnu\\+\\+11"))
      message(FATAL_ERROR
        "lld requires c++11. Clang and gcc require -std=c++0x or -std=c++11 to "
        "enter this mode. Please set CMAKE_CXX_FLAGS accordingly.")
    endif()
  endif()
else()
  message(FATAL_ERROR "The selected compiler does not support c++11 which is "
          "required to build lld.")
endif()

macro(add_lld_library name)
  llvm_process_sources(srcs ${ARGN})
  if (MSVC_IDE OR XCODE)
    string(REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list(GET split_path -1 dir)
    file(GLOB_RECURSE headers
      ../../include/lld${dir}/*.h)
    set(srcs ${srcs} ${headers})
  endif()
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind)
  endif()
  add_library(${name} ${libkind} ${srcs})
  if (LLVM_COMMON_DEPENDS)
    add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
  endif()

  target_link_libraries(${name} ${LLVM_USED_LIBS})
  llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  target_link_libraries(${name} ${LLVM_COMMON_LIBS})
  link_system_libs(${name})

  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
  set_target_properties(${name} PROPERTIES FOLDER "lld libraries")
endmacro(add_lld_library)

macro(add_lld_executable name)
  add_llvm_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "lld executables")
endmacro(add_lld_executable)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(utils)

add_subdirectory(test)

if (LLVM_INCLUDE_TESTS AND NOT LLD_BUILT_STANDALONE)
  add_subdirectory(unittests)
endif()
