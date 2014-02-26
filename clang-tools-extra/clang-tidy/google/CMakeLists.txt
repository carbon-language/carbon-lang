set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyGoogleModule
  GoogleTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  )
