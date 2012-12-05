add_subdirectory(remove-cstr-calls)
add_subdirectory(tool-template)
add_subdirectory(loop-convert)
add_subdirectory(clang-format)

# Add the common testsuite after all the tools.
add_subdirectory(test)
