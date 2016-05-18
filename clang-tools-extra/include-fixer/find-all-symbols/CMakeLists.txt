set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(findAllSymbols
  FindAllSymbols.cpp
  PragmaCommentHandler.cpp
  SymbolInfo.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
