set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp
  )

target_link_libraries(clangFrontendTool
  clangDriver
  clangFrontend
  clangRewrite
  clangCodeGen
  clangChecker
  )
