set(LLVM_NO_RTTI 1)

add_clang_library(clangSerialization
  GeneratePCH.cpp
  ASTReader.cpp
  ASTReaderDecl.cpp
  ASTReaderStmt.cpp
  ASTWriter.cpp
  ASTWriterDecl.cpp
  ASTWriterStmt.cpp
  )

add_dependencies(clangSerialization
  ClangAttrClasses
  ClangAttrList
  ClangAttrPCHRead
  ClangAttrPCHWrite
  ClangDiagnosticFrontend
  ClangDiagnosticLex
  ClangDiagnosticSema
  ClangDeclNodes
  ClangStmtNodes)
