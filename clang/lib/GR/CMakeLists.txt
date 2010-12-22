set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangLex clangAST clangFrontend clangRewrite)

add_clang_library(clangGRCore
  AggExprVisitor.cpp
  AnalysisConsumer.cpp
  AnalysisManager.cpp
  AnalyzerStatsChecker.cpp
  BasicConstraintManager.cpp
  BasicStore.cpp
  BasicValueFactory.cpp
  BugReporter.cpp
  BugReporterVisitors.cpp
  CFRefCount.cpp
  Checker.cpp
  CheckerHelpers.cpp
  Environment.cpp
  ExplodedGraph.cpp
  FlatStore.cpp
  FrontendActions.cpp
  GRBlockCounter.cpp
  GRCXXExprEngine.cpp
  GRCoreEngine.cpp
  GRExprEngine.cpp
  GRState.cpp
  HTMLDiagnostics.cpp
  ManagerRegistry.cpp
  MemRegion.cpp
  PathDiagnostic.cpp
  PlistDiagnostics.cpp
  RangeConstraintManager.cpp
  RegionStore.cpp
  SimpleConstraintManager.cpp
  SimpleSValBuilder.cpp
  Store.cpp
  SValBuilder.cpp
  SVals.cpp
  SymbolManager.cpp
  TextPathDiagnostics.cpp
  )

add_dependencies(clangGRCore ClangAttrClasses ClangAttrList ClangDeclNodes
                 ClangStmtNodes)
