# Clang version information

# Make sure that CMake reconfigures when the version changes.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/VER
  ${CMAKE_CURRENT_BINARY_DIR}/VER)

set(CLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Compute the Clang version from the contents of VER
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VER CLANG_VERSION_DATA)
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION 
  ${CLANG_VERSION_DATA})
message(STATUS "Clang version: ${CLANG_VERSION}")

# Add appropriate flags for GCC
if (CMAKE_COMPILER_IS_GNUCXX)
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
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
  if (SHARED_LIBRARY)
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
  get_system_libs(llvm_system_libs)
  if( llvm_system_libs )
    target_link_libraries(${name} ${llvm_system_libs})
  endif( llvm_system_libs )
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
    LIBRARY DESTINATION lib
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

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  )

add_definitions( -D_GNU_SOURCE )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# TODO: docs.
add_subdirectory(test)

