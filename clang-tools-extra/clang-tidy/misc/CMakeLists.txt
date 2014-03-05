set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  MiscTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  )
