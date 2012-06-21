add_clang_executable(clang-check
  ClangCheck.cpp
  )

target_link_libraries(clang-check
  clangTooling
  clangBasic
  )
