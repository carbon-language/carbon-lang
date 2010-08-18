set(LLVM_NO_RTTI 1)

add_clang_library(clangSerialization
  GeneratePCH.cpp
  PCHReader.cpp
  PCHReaderDecl.cpp
  PCHReaderStmt.cpp
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
