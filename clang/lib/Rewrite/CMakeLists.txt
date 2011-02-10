set(LLVM_USED_LIBS clangBasic clangAST clangParse clangFrontend)

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

add_dependencies(clangRewrite
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes)
