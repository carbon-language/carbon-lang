set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  ArgumentCommentCheck.cpp
  BoolPointerImplicitConversion.cpp
  MiscTidyModule.cpp
  RedundantSmartptrGet.cpp
  UseOverride.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
