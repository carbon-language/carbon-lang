set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  ArgumentCommentCheck.cpp
  BoolPointerImplicitConversion.cpp
  BracesAroundStatementsCheck.cpp
  FunctionSize.cpp
  MiscTidyModule.cpp
  RedundantSmartptrGet.cpp
  SwappedArgumentsCheck.cpp
  UndelegatedConstructor.cpp
  UnusedRAII.cpp
  UseOverride.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
