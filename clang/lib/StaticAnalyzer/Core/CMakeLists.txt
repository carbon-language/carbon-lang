set(LLVM_LINK_COMPONENTS support)

set(LLVM_USED_LIBS clangBasic clangLex clangAST clangFrontend clangRewrite)

add_clang_library(clangStaticAnalyzerCore
  AggExprVisitor.cpp
  AnalysisManager.cpp
  BasicConstraintManager.cpp
  BasicValueFactory.cpp
  BlockCounter.cpp
  BugReporter.cpp
  BugReporterVisitors.cpp
  CFRefCount.cpp
  CXXExprEngine.cpp
  Checker.cpp
  CheckerContext.cpp
  CheckerHelpers.cpp
  CheckerManager.cpp
  CoreEngine.cpp
  Environment.cpp
  ExplodedGraph.cpp
  ExprEngine.cpp
  HTMLDiagnostics.cpp
  MemRegion.cpp
  ObjCMessage.cpp
  PathDiagnostic.cpp
  PlistDiagnostics.cpp
  ProgramState.cpp
  RangeConstraintManager.cpp
  RegionStore.cpp
  SValBuilder.cpp
  SVals.cpp
  SimpleConstraintManager.cpp
  SimpleSValBuilder.cpp
  Store.cpp
  SymbolManager.cpp
  TextPathDiagnostics.cpp
  )

add_dependencies(clangStaticAnalyzerCore ClangAttrClasses ClangAttrList ClangDeclNodes
                 ClangStmtNodes)
