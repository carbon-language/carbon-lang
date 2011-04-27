set(LLVM_USED_LIBS clangTooling clangBasic)

add_clang_executable(clang-check
  ClangCheck.cpp
  )

