set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangRewriteFrontend
  FixItRewriter.cpp
  FrontendActions.cpp
  HTMLPrint.cpp
  InclusionRewriter.cpp
  RewriteMacros.cpp
  RewriteModernObjC.cpp
  RewriteObjC.cpp
  RewriteTest.cpp
  )

target_link_libraries(clangRewriteFrontend
  clangAST
  clangBasic
  clangEdit
  clangFrontend
  clangLex
  clangRewriteCore
  )
