set(LLVM_NO_RTTI 1)

add_clang_library(clangSema
  IdentifierResolver.cpp
  JumpDiagnostics.cpp
  ParseAST.cpp
  Sema.cpp
  SemaAccess.cpp
  SemaAttr.cpp
  SemaCXXCast.cpp
  SemaCXXScopeSpec.cpp
  SemaChecking.cpp
  SemaDecl.cpp
  SemaDeclAttr.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprObjC.cpp
  SemaInherit.cpp
  SemaInit.cpp
  SemaLookup.cpp
  SemaOverload.cpp
  SemaStmt.cpp
  SemaTemplate.cpp
  SemaTemplateDeduction.cpp
  SemaTemplateInstantiate.cpp
  SemaTemplateInstantiateDecl.cpp
  SemaTemplateInstantiateStmt.cpp
  SemaType.cpp
  )

add_dependencies(clangSema ClangDiagnosticSema)
