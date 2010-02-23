set(LLVM_NO_RTTI 1)

add_clang_library(clangAnalysis
  AnalysisContext.cpp
  CFG.cpp
  LiveVariables.cpp
  PrintfFormatString.cpp
  ReachableCode.cpp
  UninitializedValues.cpp
  )

add_dependencies(clangAnalysis ClangDiagnosticAnalysis)
