set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangRewriteCore
  DeltaTree.cpp
  HTMLRewrite.cpp
  RewriteRope.cpp
  Rewriter.cpp
  TokenRewriter.cpp
  )

target_link_libraries(clangRewriteCore
  clangAST
  clangBasic
  clangLex
  )
