set(LLVM_LINK_COMPONENTS support)
SET(LLVM_USED_LIBS clangBasic clangFrontend clangAST clangRewrite)

add_clang_library(clangTooling
  CompilationDatabase.cpp
  Refactoring.cpp
  Tooling.cpp
  ArgumentsAdjusters.cpp
  )
