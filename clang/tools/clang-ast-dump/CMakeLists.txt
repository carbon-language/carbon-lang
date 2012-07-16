add_clang_executable(clang-ast-dump
  ClangASTDump.cpp
  )

target_link_libraries(clang-ast-dump
  clangTooling
  clangBasic
  )
