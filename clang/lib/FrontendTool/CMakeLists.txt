add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )

add_dependencies(clangFrontendTool
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangDriverOptions
  )

target_link_libraries(clangFrontendTool
  clangDriver
  clangFrontend
  clangRewriteCore
  clangRewriteFrontend
  clangCodeGen
  clangARCMigrate
  )

if(CLANG_ENABLE_STATIC_ANALYZER)
  target_link_libraries(clangFrontendTool
    clangStaticAnalyzerFrontend
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    )
endif()
