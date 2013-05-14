add_subdirectory(Dynamic)

set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangASTMatchers
  ASTMatchFinder.cpp
  ASTMatchersInternal.cpp
  )

add_dependencies(clangASTMatchers
  ClangAttrClasses
  ClangAttrImpl
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )

target_link_libraries(clangASTMatchers
  clangBasic
  clangAST
  )
