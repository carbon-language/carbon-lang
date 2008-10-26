set(LLVM_NO_RTTI 1)

add_clang_library(clangSema
  IdentifierResolver.cpp
  ParseAST.cpp
  SemaChecking.cpp
  Sema.cpp
  SemaDeclAttr.cpp
  SemaDecl.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprObjC.cpp
  SemaInherit.cpp
  SemaInit.cpp
  SemaOverload.cpp
  SemaStmt.cpp
  SemaType.cpp
  )
