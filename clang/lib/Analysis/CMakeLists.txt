set(LLVM_NO_RTTI 1)

add_clang_library(clangAnalysis
  BasicConstraintManager.cpp
  BasicObjCFoundationChecks.cpp
  BasicStore.cpp
  BasicValueFactory.cpp
  BugReporter.cpp
  CFG.cpp
  CFRefCount.cpp
  CallGraph.cpp
  CheckDeadStores.cpp
  CheckNSError.cpp
  CheckObjCDealloc.cpp
  CheckObjCInstMethSignature.cpp
  CheckObjCUnusedIVars.cpp
  Environment.cpp
  ExplodedGraph.cpp
  GRBlockCounter.cpp
  GRCoreEngine.cpp
  GRExprEngine.cpp
  GRExprEngineInternalChecks.cpp
  GRState.cpp
  LiveVariables.cpp
  MemRegion.cpp
  PathDiagnostic.cpp
  RangeConstraintManager.cpp
  RegionStore.cpp
  SVals.cpp
  SValuator.cpp
  SimpleConstraintManager.cpp
  SimpleSValuator.cpp
  Store.cpp
  SymbolManager.cpp
  UninitializedValues.cpp
  ValueManager.cpp
  )

add_dependencies(clangAnalysis ClangDiagnosticAnalysis)
