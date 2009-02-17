set(LLVM_NO_RTTI 1)

add_clang_library(clangSema
  IdentifierResolver.cpp
  ParseAST.cpp
  Sema.cpp
  SemaAttr.cpp
  SemaChecking.cpp
  SemaCXXScopeSpec.cpp
  SemaDeclAttr.cpp
  SemaDecl.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprObjC.cpp
  SemaInherit.cpp
  SemaInit.cpp
  SemaLookup.cpp
  SemaNamedCast.cpp
  SemaOverload.cpp
  SemaStmt.cpp
  SemaTemplate.cpp
  SemaType.cpp
  )
