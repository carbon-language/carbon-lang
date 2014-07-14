set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyGoogleModule
  AvoidCStyleCastsCheck.cpp
  ExplicitConstructorCheck.cpp
  GoogleTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
