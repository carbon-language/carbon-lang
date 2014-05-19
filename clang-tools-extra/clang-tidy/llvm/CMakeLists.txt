set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyLLVMModule
  IncludeOrderCheck.cpp
  LLVMTidyModule.cpp
  NamespaceCommentCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
