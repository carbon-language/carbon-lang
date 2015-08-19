set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyReadabilityModule
  BracesAroundStatementsCheck.cpp
  ContainerSizeEmptyCheck.cpp
  ElseAfterReturnCheck.cpp
  FunctionSizeCheck.cpp
  IdentifierNamingCheck.cpp
  NamedParameterCheck.cpp
  NamespaceCommentCheck.cpp
  ReadabilityTidyModule.cpp
  RedundantStringCStrCheck.cpp
  RedundantSmartptrGetCheck.cpp
  ShrinkToFitCheck.cpp
  SimplifyBooleanExprCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTooling
  )
