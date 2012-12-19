include(LLVMParseArguments)

# Compile a source into an object file with just-built Clang using
# a provided compile flags and dependenices.
# clang_compile(<object> <source>
#               CFLAGS <list of compile flags>
#               DEPS <list of dependencies>)
macro(clang_compile object_file source)
  parse_arguments(SOURCE "CFLAGS;DEPS" "" ${ARGN})
  get_filename_component(source_rpath ${source} REALPATH)
  add_custom_command(
    OUTPUT ${object_file}
    COMMAND clang ${SOURCE_CFLAGS} -c -o "${object_file}" ${source_rpath}
    MAIN_DEPENDENCY ${source}
    DEPENDS clang ${SOURCE_DEPS})
endmacro()
