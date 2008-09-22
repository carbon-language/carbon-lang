
macro(add_partially_linked_object lib)
  if( MSVC )
    add_llvm_library( ${lib} ${ARGN})
  else( MSVC )
    set(pll ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${lib}.o)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/temp_lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/temp_lib)
    add_library( ${lib} STATIC ${ARGN})
    add_custom_command(OUTPUT ${pll}
      MESSAGE "Building ${lib}.o..."
      DEPENDS ${lib}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/temp_lib
      COMMAND ar x ${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}
      COMMAND ld -r *${CMAKE_CXX_OUTPUT_EXTENSION} -o ${pll}
      COMMAND rm -f *${CMAKE_CXX_OUTPUT_EXTENSION}
      )
    add_custom_target(${lib}_pll ALL DEPENDS ${pll})
    set( llvm_libs ${llvm_libs} ${pll} PARENT_SCOPE)
  endif( MSVC )
endmacro(add_partially_linked_object lib)
