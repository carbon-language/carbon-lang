set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangAST)

add_clang_library(clangStaticAnalyzerCheckers
  AdjustedReturnValueChecker.cpp
  AnalysisConsumer.cpp
  ArrayBoundChecker.cpp
  ArrayBoundCheckerV2.cpp
  AttrNonNullChecker.cpp
  BasicObjCFoundationChecks.cpp
  BuiltinFunctionChecker.cpp
  CStringChecker.cpp
  CallAndMessageChecker.cpp
  CastSizeChecker.cpp
  CastToStructChecker.cpp
  CheckObjCDealloc.cpp
  CheckObjCInstMethSignature.cpp
  CheckSecuritySyntaxOnly.cpp
  CheckSizeofPointer.cpp
  ChrootChecker.cpp
  DeadStoresChecker.cpp
  DereferenceChecker.cpp
  DivZeroChecker.cpp
  ExprEngine.cpp
  ExprEngineExperimentalChecks.cpp
  FixedAddressChecker.cpp
  FrontendActions.cpp
  IdempotentOperationChecker.cpp
  LLVMConventionsChecker.cpp
  MacOSXAPIChecker.cpp
  MallocChecker.cpp
  NSAutoreleasePoolChecker.cpp
  NSErrorChecker.cpp
  NoReturnFunctionChecker.cpp
  OSAtomicChecker.cpp
  ObjCAtSyncChecker.cpp
  ObjCSelfInitChecker.cpp
  ObjCUnusedIVarsChecker.cpp
  PointerArithChecker.cpp
  PointerSubChecker.cpp
  PthreadLockChecker.cpp
  ReturnPointerRangeChecker.cpp
  ReturnUndefChecker.cpp
  StackAddrLeakChecker.cpp
  StreamChecker.cpp
  UndefBranchChecker.cpp
  UndefCapturedBlockVarChecker.cpp
  UndefResultChecker.cpp
  UndefinedArraySubscriptChecker.cpp
  UndefinedAssignmentChecker.cpp
  UnixAPIChecker.cpp
  UnreachableCodeChecker.cpp
  VLASizeChecker.cpp
  )

add_dependencies(clangStaticAnalyzerCore ClangAttrClasses ClangAttrList ClangDeclNodes
                 ClangStmtNodes)
