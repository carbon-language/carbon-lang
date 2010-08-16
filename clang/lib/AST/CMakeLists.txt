set(LLVM_NO_RTTI 1)

add_clang_library(clangAST
  APValue.cpp
  ASTConsumer.cpp
  ASTContext.cpp
  ASTDiagnostic.cpp
  ASTImporter.cpp
  AttrImpl.cpp
  CXXInheritance.cpp
  Decl.cpp
  DeclarationName.cpp
  DeclBase.cpp
  DeclCXX.cpp
  DeclFriend.cpp
  DeclGroup.cpp
  DeclObjC.cpp
  DeclPrinter.cpp
  DeclTemplate.cpp
  Expr.cpp
  ExprClassification.cpp
  ExprConstant.cpp
  ExprCXX.cpp
  FullExpr.cpp
  InheritViz.cpp
  ItaniumCXXABI.cpp
  MicrosoftCXXABI.cpp
  NestedNameSpecifier.cpp
  ParentMap.cpp
  RecordLayout.cpp
  RecordLayoutBuilder.cpp
  Stmt.cpp
  StmtDumper.cpp
  StmtIterator.cpp
  StmtPrinter.cpp
  StmtProfile.cpp
  StmtViz.cpp
  TemplateBase.cpp
  TemplateName.cpp
  Type.cpp
  TypeLoc.cpp
  TypePrinter.cpp
  )

add_dependencies(clangAST ClangARMNeon ClangAttrClasses ClangAttrList 
                 ClangDiagnosticAST ClangDeclNodes ClangStmtNodes)
