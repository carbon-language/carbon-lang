set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  ArgumentCommentCheck.cpp
  BoolPointerImplicitConversion.cpp
  MiscTidyModule.cpp
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
  # Some readability checks are currently registered in the misc module.
  clangTidyReadability
  )
