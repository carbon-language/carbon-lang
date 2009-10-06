set(LLVM_NO_RTTI 1)

add_clang_library(clangAST
  APValue.cpp
  ASTConsumer.cpp
  ASTContext.cpp
  CXXInheritance.cpp
  Decl.cpp
  DeclBase.cpp
  DeclCXX.cpp
  DeclGroup.cpp
  DeclObjC.cpp
  DeclPrinter.cpp
  DeclTemplate.cpp
  DeclarationName.cpp
  Expr.cpp
  ExprCXX.cpp
  ExprConstant.cpp
  InheritViz.cpp
  NestedNameSpecifier.cpp
  ParentMap.cpp
  RecordLayoutBuilder.cpp
  Stmt.cpp
  StmtDumper.cpp
  StmtIterator.cpp
  StmtPrinter.cpp
  StmtProfile.cpp
  StmtViz.cpp
  TemplateName.cpp
  Type.cpp
  TypeLoc.cpp
  )

add_dependencies(clangAST ClangDiagnosticAST)
