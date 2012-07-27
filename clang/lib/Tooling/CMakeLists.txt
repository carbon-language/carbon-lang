set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTooling
  ArgumentsAdjusters.cpp
  CommandLineClangTool.cpp
  CompilationDatabase.cpp
  Refactoring.cpp
  RefactoringCallbacks.cpp
  Tooling.cpp
  )

add_dependencies(clangTooling
  ClangDiagnosticCommon
  )

target_link_libraries(clangTooling
  clangBasic
  clangFrontend
  clangAST
  clangASTMatchers
  clangRewrite
  )
