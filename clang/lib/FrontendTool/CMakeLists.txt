add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )

add_dependencies(clangFrontendTool
  ClangDiagnosticFrontend
  )

target_link_libraries(clangFrontendTool
  clangDriver
  clangFrontend
  clangRewrite
  clangCodeGen
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  clangARCMigrate
  )
