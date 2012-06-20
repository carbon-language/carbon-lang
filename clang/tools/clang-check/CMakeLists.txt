set(LLVM_USED_LIBS clangTooling clangBasic clangComments)

add_clang_executable(clang-check
  ClangCheck.cpp
  )
