set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTooling
  ArgumentsAdjusters.cpp
  CommonOptionsParser.cpp
  CompilationDatabase.cpp
  FileMatchTrie.cpp
  JSONCompilationDatabase.cpp
  Refactoring.cpp
  RefactoringCallbacks.cpp
  Tooling.cpp
  )

target_link_libraries(clangTooling
  clangAST
  clangASTMatchers
  clangBasic
  clangDriver
  clangFrontend
  clangLex
  clangRewriteCore
  )
