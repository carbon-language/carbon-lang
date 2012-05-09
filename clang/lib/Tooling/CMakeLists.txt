set(LLVM_LINK_COMPONENTS support)
SET(LLVM_USED_LIBS clangBasic clangFrontend clangAST)

add_clang_library(clangTooling
  CompilationDatabase.cpp
  Tooling.cpp
  ArgumentsAdjusters.cpp
  )
