set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangFormat
  UnwrappedLineParser.cpp
  Format.cpp
  )

add_dependencies(clangFormat
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangStmtNodes
  )

target_link_libraries(clangFormat
  clangBasic
  clangFrontend
  clangAST
  clangASTMatchers
  clangRewriteCore
  clangRewriteFrontend
  )
