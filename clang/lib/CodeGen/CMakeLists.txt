set(LLVM_NO_RTTI 1)

add_clang_library(clangCodeGen
  CGBuiltin.cpp
  CGBlocks.cpp
  CGCall.cpp
  CGCXX.cpp
  CGDebugInfo.cpp
  CGDecl.cpp
  CGExprAgg.cpp
  CGExprComplex.cpp
  CGExprConstant.cpp
  CGExpr.cpp
  CGExprScalar.cpp
  CGObjC.cpp
  CGObjCGNU.cpp
  CGObjCMac.cpp
  CGStmt.cpp
  CodeGenFunction.cpp
  CodeGenModule.cpp
  CodeGenTypes.cpp
  Mangle.cpp
  ModuleBuilder.cpp
  )
