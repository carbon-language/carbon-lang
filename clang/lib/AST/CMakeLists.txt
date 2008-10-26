set(LLVM_NO_RTTI 1)

add_clang_library(clangAST
  ASTConsumer.cpp
  ASTContext.cpp
  Builtins.cpp
  CFG.cpp
  DeclBase.cpp
  Decl.cpp
  DeclCXX.cpp
  DeclGroup.cpp
  DeclObjC.cpp
  DeclSerialization.cpp
  ExprConstant.cpp
  Expr.cpp
  ExprCXX.cpp
  InheritViz.cpp
  ParentMap.cpp
  Stmt.cpp
  StmtDumper.cpp
  StmtIterator.cpp
  StmtPrinter.cpp
  StmtSerialization.cpp
  StmtViz.cpp
  TranslationUnit.cpp
  Type.cpp
  TypeSerialization.cpp
  )
