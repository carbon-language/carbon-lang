set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyReadabilityModule
  BracesAroundStatementsCheck.cpp
  FunctionSize.cpp
  NamespaceCommentCheck.cpp
  ReadabilityTidyModule.cpp
  RedundantSmartptrGet.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTooling
  )
