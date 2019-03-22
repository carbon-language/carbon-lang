set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyOpenMPModule
  OpenMPTidyModule.cpp
  UseDefaultNoneCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  )
