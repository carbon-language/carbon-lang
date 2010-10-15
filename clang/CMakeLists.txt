# Clang version information

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
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
endif ()

if (APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif ()

macro(add_clang_library name)
  set(srcs ${ARGN})
  if(MSVC_IDE OR XCODE)
    file( GLOB_RECURSE headers *.h *.td *.def)
    set(srcs ${srcs} ${headers})
    string( REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list( GET split_path -1 dir)
    file( GLOB_RECURSE headers
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
  set(srcs ${ARGN})
  if(MSVC_IDE)
    file( GLOB_RECURSE headers *.h *.td *.def)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE)
  add_llvm_executable( ${name} ${srcs} )
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

add_definitions( -D_GNU_SOURCE )

option(CLANG_BUILD_EXAMPLES "Build CLANG example programs." OFF)
if(CLANG_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# TODO: docs.
add_subdirectory(test)

