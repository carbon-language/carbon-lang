macro(add_clang_library name)
  add_library( ${name} ${ARGN} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  install(TARGETS ${name}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
endmacro(add_clang_library)

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  install(TARGETS ${name}
    RUNTIME DESTINATION bin)
endmacro(add_clang_executable)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

add_definitions( -D_GNU_SOURCE )

add_subdirectory(lib)
add_subdirectory(Driver)

# TODO: docs.
