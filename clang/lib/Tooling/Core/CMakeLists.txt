set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingCore
  Lookup.cpp
  Replacement.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangRewrite
  )
