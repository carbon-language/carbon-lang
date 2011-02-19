set(LLVM_TARGET_DEFINITIONS Checkers.td)
tablegen(Checkers.inc
         -gen-clang-sa-checkers
         -I ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
add_custom_target(ClangSACheckers
  DEPENDS Checkers.inc)

set(LLVM_USED_LIBS clangBasic clangAST)

add_clang_library(clangStaticAnalyzerCheckers
  AdjustedReturnValueChecker.cpp
  AnalyzerStatsChecker.cpp
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
  ClangSACheckerProvider.cpp
  DeadStoresChecker.cpp
  DebugCheckers.cpp
  DereferenceChecker.cpp
  DivZeroChecker.cpp
  ExprEngine.cpp
  ExperimentalChecks.cpp
  FixedAddressChecker.cpp
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

add_dependencies(clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes
  ClangSACheckers
  )
