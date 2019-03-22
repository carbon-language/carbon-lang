set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyOpenMPModule
  OpenMPTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  )
