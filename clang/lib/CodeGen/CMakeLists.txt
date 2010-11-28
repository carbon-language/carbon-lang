set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangAST clangFrontend)

add_clang_library(clangCodeGen
  BackendUtil.cpp
  CGBlocks.cpp
  CGBuiltin.cpp
  CGCall.cpp
  CGClass.cpp
  CGCXX.cpp
  CGCXXABI.cpp
  CGDebugInfo.cpp
  CGDecl.cpp
  CGDeclCXX.cpp
  CGException.cpp
  CGExpr.cpp
  CGExprAgg.cpp
  CGExprComplex.cpp
  CGExprConstant.cpp
  CGExprCXX.cpp
  CGExprScalar.cpp
  CGObjC.cpp
  CGObjCGNU.cpp
  CGObjCMac.cpp
  CGRecordLayoutBuilder.cpp
  CGRTTI.cpp
  CGStmt.cpp
  CGTemporaries.cpp
  CGVTables.cpp
  CGVTT.cpp
  CodeGenAction.cpp
  CodeGenFunction.cpp
  CodeGenModule.cpp
  CodeGenTBAA.cpp
  CodeGenTypes.cpp
  ItaniumCXXABI.cpp
  Mangle.cpp
  MicrosoftCXXABI.cpp
  ModuleBuilder.cpp
  TargetInfo.cpp
  )

add_dependencies(clangCodeGen ClangAttrClasses ClangAttrList ClangDeclNodes
                 ClangStmtNodes)
