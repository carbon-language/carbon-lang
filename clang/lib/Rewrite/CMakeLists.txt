set(LLVM_NO_RTTI 1)

add_clang_library(clangRewrite
  DeltaTree.cpp
  FixItRewriter.cpp
  FrontendActions.cpp
  HTMLPrint.cpp
  HTMLRewrite.cpp
  RewriteMacros.cpp
  RewriteObjC.cpp
  RewriteRope.cpp
  RewriteTest.cpp
  Rewriter.cpp
  TokenRewriter.cpp
  )

add_dependencies(clangBasic
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes)
