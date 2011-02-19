set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangLex clangAST clangFrontend clangRewrite)

include_directories( ${CMAKE_CURRENT_BINARY_DIR}/../Checkers )

add_clang_library(clangStaticAnalyzerFrontend
  AnalysisConsumer.cpp
  CheckerRegistration.cpp
  FrontendActions.cpp
  )

add_dependencies(clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNode
  )
