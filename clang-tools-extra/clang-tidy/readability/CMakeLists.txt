set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyReadability
  BracesAroundStatementsCheck.cpp
  FunctionSize.cpp
  NamespaceCommentCheck.cpp
  RedundantSmartptrGet.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTooling
  )
