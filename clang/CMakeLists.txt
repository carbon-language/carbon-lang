# Clang version information
# FIXME: get from a file called ./VER
set(CLANG_VERSION 1.1)

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
  add_library( ${name} ${srcs} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
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

