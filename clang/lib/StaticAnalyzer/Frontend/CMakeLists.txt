set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangLex clangAST clangFrontend clangRewrite)

add_clang_library(clangStaticAnalyzerFrontend
  AnalysisConsumer.cpp
  CheckerRegistration.cpp
  FrontendActions.cpp
  )

add_dependencies(clangStaticAnalyzerCheckers clangStaticAnalyzerCore ClangAttrClasses ClangAttrList ClangDeclNodes
                 ClangStmtNodes)
