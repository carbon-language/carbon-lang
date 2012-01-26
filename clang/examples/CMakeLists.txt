if(NOT CLANG_BUILD_EXAMPLES)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_subdirectory(clang-interpreter)
add_subdirectory(PrintFunctionNames)
