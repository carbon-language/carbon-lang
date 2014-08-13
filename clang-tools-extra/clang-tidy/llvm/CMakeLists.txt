set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyLLVMModule
  HeaderGuardCheck.cpp
  IncludeOrderCheck.cpp
  LLVMTidyModule.cpp
  NamespaceCommentCheck.cpp
  TwineLocalCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  )
