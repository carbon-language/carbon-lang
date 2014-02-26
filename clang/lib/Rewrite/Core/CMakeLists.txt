set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangRewriteCore
  DeltaTree.cpp
  HTMLRewrite.cpp
  RewriteRope.cpp
  Rewriter.cpp
  TokenRewriter.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  )
