set(LLVM_USED_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangEdit
  clangLex
  clangComments
  )

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
  SemaCXXScopeSpec.cpp
  SemaCast.cpp
  SemaChecking.cpp
  SemaCodeComplete.cpp
  SemaConsumer.cpp
  SemaDecl.cpp
  SemaDeclAttr.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExceptionSpec.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprMember.cpp
  SemaExprObjC.cpp
  SemaFixItUtils.cpp
  SemaInit.cpp
  SemaLambda.cpp
  SemaLookup.cpp
  SemaObjCProperty.cpp
  SemaOverload.cpp
  SemaPseudoObject.cpp
  SemaStmt.cpp
  SemaStmtAttr.cpp
  SemaTemplate.cpp
  SemaTemplateDeduction.cpp
  SemaTemplateInstantiate.cpp
  SemaTemplateInstantiateDecl.cpp
  SemaTemplateVariadic.cpp
  SemaType.cpp
  TargetAttributesSema.cpp
  )

add_dependencies(clangSema ClangARMNeon ClangAttrClasses ClangAttrList 
                 ClangDiagnosticSema ClangDeclNodes ClangStmtNodes 
                 ClangAttrTemplateInstantiate ClangAttrParsedAttrList 
				 ClangAttrParsedAttrKinds)


