set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTooling
  CommandLineClangTool.cpp
  CompilationDatabase.cpp
  Refactoring.cpp
  Tooling.cpp
  ArgumentsAdjusters.cpp
  )

target_link_libraries(clangTooling
  clangBasic
  clangFrontend
  clangAST
  clangRewrite
  )
