set(LLVM_LINK_COMPONENTS support)

set(LLVM_USED_LIBS clangBasic clangLex)

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
  DumpXML.cpp
  Expr.cpp
  ExprClassification.cpp
  ExprConstant.cpp
  ExprCXX.cpp
  ExternalASTSource.cpp
  InheritViz.cpp
  ItaniumCXXABI.cpp
  ItaniumMangle.cpp
  Mangle.cpp
  MicrosoftCXXABI.cpp
  MicrosoftMangle.cpp
  NestedNameSpecifier.cpp
  ParentMap.cpp
  RecordLayout.cpp
  RecordLayoutBuilder.cpp
  SelectorLocationsKind.cpp
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
  VTableBuilder.cpp
  VTTBuilder.cpp
  )

add_dependencies(clangAST ClangARMNeon ClangAttrClasses ClangAttrList 
                 ClangAttrImpl ClangDiagnosticAST ClangDeclNodes ClangStmtNodes)
