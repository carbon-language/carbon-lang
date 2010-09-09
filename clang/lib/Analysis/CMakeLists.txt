set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangAST clangIndex)

add_clang_library(clangAnalysis
  AnalysisContext.cpp
  CFG.cpp
  CFGStmtMap.cpp
  FormatString.cpp
  LiveVariables.cpp
  PrintfFormatString.cpp
  PseudoConstantAnalysis.cpp
  ReachableCode.cpp
  ScanfFormatString.cpp
  UninitializedValues.cpp
  )

add_dependencies(clangAnalysis ClangAttrClasses ClangAttrList
                 ClangDiagnosticAnalysis ClangDeclNodes ClangStmtNodes)
