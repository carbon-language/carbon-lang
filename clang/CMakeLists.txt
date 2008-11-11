macro(add_clang_library name)
  set(srcs ${ARGN})
  if(MSVC_IDE)
    file( GLOB_RECURSE headers *.h)
    set(srcs ${srcs} ${headers})
    string( REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list( GET split_path -1 dir)
    file( GLOB_RECURSE headers ../../include/clang${dir}/*.h)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE)
  add_library( ${name} ${srcs} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  install(TARGETS ${name}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
endmacro(add_clang_library)

macro(add_clang_executable name)
  set(srcs ${ARGN})
  if(MSVC_IDE)
    file( GLOB_RECURSE headers *.h)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE)
  add_llvm_executable( ${name} ${srcs} )
  install(TARGETS ${name}
    RUNTIME DESTINATION bin)
endmacro(add_clang_executable)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  )

add_definitions( -D_GNU_SOURCE )

add_subdirectory(lib)
add_subdirectory(Driver)

# TODO: docs.
