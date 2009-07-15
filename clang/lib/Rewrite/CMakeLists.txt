set(LLVM_NO_RTTI 1)

add_clang_library(clangRewrite
  DeltaTree.cpp
  HTMLRewrite.cpp
  RewriteRope.cpp
  Rewriter.cpp
  TokenRewriter.cpp
  )
