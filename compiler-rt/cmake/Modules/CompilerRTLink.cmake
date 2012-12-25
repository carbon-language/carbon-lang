include(LLVMParseArguments)

# Link a shared library with just-built Clang.
# clang_link_shared(<output.so>
#                   OBJECTS <list of input objects>
#                   LINKFLAGS <list of link flags>
#                   DEPS <list of dependencies>)
macro(clang_link_shared so_file)
  parse_arguments(SOURCE "OBJECTS;LINKFLAGS;DEPS" "" ${ARGN})
  add_custom_command(
    OUTPUT ${so_file}
    COMMAND clang -o "${so_file}" -shared ${SOURCE_LINKFLAGS} ${SOURCE_OBJECTS}
    DEPENDS clang ${SOURCE_DEPS})
endmacro()
