# TODO: This must need some dependencies, but it builds fine without them.
#set(LLVM_USED_LIBS ???)

add_clang_library(clangSerialization
  ASTCommon.h
  ASTReaderInternals.h
  ASTCommon.cpp
  ASTReader.cpp
  ASTReaderDecl.cpp
  ASTReaderStmt.cpp
  ASTWriter.cpp
  ASTWriterDecl.cpp
  ASTWriterStmt.cpp
  ChainedIncludesSource.cpp
  GeneratePCH.cpp
  Module.cpp
  ModuleManager.cpp
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
