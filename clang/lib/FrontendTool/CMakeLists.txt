set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangDriver clangFrontend clangRewrite clangCodeGen 
    clangStaticAnalyzerCheckers clangStaticAnalyzerCore)

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )
