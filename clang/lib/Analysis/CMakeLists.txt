set(LLVM_NO_RTTI 1)

add_clang_library(clangAnalysis
  BasicConstraintManager.cpp
  BasicObjCFoundationChecks.cpp
  BasicStore.cpp
  BasicValueFactory.cpp
  BugReporter.cpp
  CFRefCount.cpp
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
  GRSimpleVals.cpp
  GRState.cpp
  GRTransferFuncs.cpp
  LiveVariables.cpp
  MemRegion.cpp
  PathDiagnostic.cpp
  RangeConstraintManager.cpp
  RegionStore.cpp
  SimpleConstraintManager.cpp
  Store.cpp
  SVals.cpp
  SymbolManager.cpp
  UninitializedValues.cpp
  )

add_dependencies(clangAnalysis ClangDiagnosticAnalysis)
