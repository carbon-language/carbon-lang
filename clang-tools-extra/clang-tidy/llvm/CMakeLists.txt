set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyLLVMModule
  LLVMTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
