include(LLVMParseArguments)

# Compile a source into an object file with COMPILER_RT_TEST_COMPILER using
# a provided compile flags and dependenices.
# clang_compile(<object> <source>
#               CFLAGS <list of compile flags>
#               DEPS <list of dependencies>)
macro(clang_compile object_file source)
  parse_arguments(SOURCE "CFLAGS;DEPS" "" ${ARGN})
  get_filename_component(source_rpath ${source} REALPATH)
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND SOURCE_DEPS clang)
  endif()
  string(REGEX MATCH "[.](cc|cpp)$" is_cxx ${source_rpath})
  if(is_cxx)
    string(REPLACE " " ";" global_flags "${CMAKE_CXX_FLAGS}")
  else()
    string(REPLACE " " ";" global_flags "${CMAKE_C_FLAGS}")
  endif()
  # Ignore unknown warnings. CMAKE_CXX_FLAGS may contain GCC-specific options
  # which are not supported by Clang.
  list(APPEND global_flags -Wno-unknown-warning-option)
  set(compile_flags ${global_flags} ${SOURCE_CFLAGS})
  add_custom_command(
    OUTPUT ${object_file}
    # MSVS CL doesn't allow a space between -Fo and the object file name.
    COMMAND ${COMPILER_RT_TEST_COMPILER} ${compile_flags} -c
            ${COMPILER_RT_TEST_COMPILER_OBJ}"${object_file}"
            ${source_rpath}
    MAIN_DEPENDENCY ${source}
    DEPENDS ${SOURCE_DEPS})
endmacro()
