SET(LLVM_USED_LIBS clangBasic clangFrontend clangAST)

add_clang_library(clangTooling
  Tooling.cpp
  )
