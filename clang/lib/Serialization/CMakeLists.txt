set(LLVM_NO_RTTI 1)

add_clang_library(clangSerialization
  GeneratePCH.cpp
  ASTCommon.cpp
  ASTReader.cpp
  ASTReaderDecl.cpp
  ASTReaderStmt.cpp
  ASTWriter.cpp
  ASTWriterDecl.cpp
  ASTWriterStmt.cpp
  )

target_link_libraries(clangSerialization
  clangFrontend
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
  ClangStmtNodes
  )
