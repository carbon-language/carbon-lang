set(LLVM_USED_LIBS clangDriver clangFrontend clangRewrite clangCodeGen 
    clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers clangStaticAnalyzerCore)

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )
