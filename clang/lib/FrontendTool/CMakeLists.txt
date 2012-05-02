set(LLVM_USED_LIBS clangDriver clangFrontend clangRewrite clangCodeGen
    clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers clangStaticAnalyzerCore
    clangARCMigrate)

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )

add_dependencies(clangFrontendTool
  ClangDiagnosticFrontend)
