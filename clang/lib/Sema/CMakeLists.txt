set(LLVM_NO_RTTI 1)

add_clang_library(clangSema
  AnalysisBasedWarnings.cpp
  CodeCompleteConsumer.cpp
  IdentifierResolver.cpp
  JumpDiagnostics.cpp
  ParseAST.cpp
  Sema.cpp
  SemaAccess.cpp
  SemaAttr.cpp
  SemaCXXCast.cpp
  SemaCXXScopeSpec.cpp
  SemaChecking.cpp
  SemaCodeComplete.cpp
  SemaDecl.cpp
  SemaDeclAttr.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExceptionSpec.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprObjC.cpp
  SemaInit.cpp
  SemaLookup.cpp
  SemaObjCProperty.cpp
  SemaOverload.cpp
  SemaStmt.cpp
  SemaTemplate.cpp
  SemaTemplateDeduction.cpp
  SemaTemplateInstantiate.cpp
  SemaTemplateInstantiateDecl.cpp
  SemaType.cpp
  TargetAttributesSema.cpp
  )

add_dependencies(clangSema ClangDiagnosticSema)
