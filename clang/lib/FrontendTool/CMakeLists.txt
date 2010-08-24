set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )
