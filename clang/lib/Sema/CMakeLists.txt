set(LLVM_USED_LIBS clangBasic clangAST clangLex clangAnalysis)

add_clang_library(clangSema
  AnalysisBasedWarnings.cpp
  AttributeList.cpp
  CodeCompleteConsumer.cpp
  DeclSpec.cpp
  DelayedDiagnostic.cpp
  IdentifierResolver.cpp
  JumpDiagnostics.cpp
  Scope.cpp
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
  SemaExprMember.cpp
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
  SemaTemplateVariadic.cpp
  SemaType.cpp
  TargetAttributesSema.cpp
  )

add_dependencies(clangSema ClangARMNeon ClangAttrClasses ClangAttrList 
                 ClangDiagnosticSema ClangDeclNodes ClangStmtNodes)
