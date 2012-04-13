set(LLVM_USED_LIBS clangBasic clangAST)

add_clang_library(clangAnalysis
  AnalysisDeclContext.cpp
  CallGraph.cpp
  CFG.cpp
  CFGReachabilityAnalysis.cpp
  CFGStmtMap.cpp
  CocoaConventions.cpp
  Dominators.cpp
  FormatString.cpp
  LiveVariables.cpp
  PostOrderCFGView.cpp
  PrintfFormatString.cpp
  ProgramPoint.cpp
  PseudoConstantAnalysis.cpp
  ReachableCode.cpp
  ScanfFormatString.cpp
  ThreadSafety.cpp
  UninitializedValues.cpp
  )

add_dependencies(clangAnalysis ClangAttrClasses ClangAttrList
                 ClangDiagnosticAnalysis ClangDeclNodes ClangStmtNodes)
