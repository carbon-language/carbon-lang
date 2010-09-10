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

target_link_libraries(clangRewrite
  clangBasic
  clangAST
  clangParse
  clangFrontend
  )

add_dependencies(clangRewrite
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes
  )
