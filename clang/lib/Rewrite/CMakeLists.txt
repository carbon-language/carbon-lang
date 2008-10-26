set(LLVM_NO_RTTI 1)

add_clang_library(clangRewrite
  DeltaTree.cpp
  HTMLRewrite.cpp
  Rewriter.cpp
  RewriteRope.cpp
  TokenRewriter.cpp
  )
