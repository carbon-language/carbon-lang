set(LLVM_NO_RTTI 1)

add_clang_library(clangSerialization
  GeneratePCH.cpp
  PCHReader.cpp
  PCHReaderDecl.cpp
  PCHReaderStmt.cpp
  PCHWriter.cpp
  PCHWriterDecl.cpp
  PCHWriterStmt.cpp
  )

add_dependencies(clangSerialization
  ClangAttrClasses
  ClangAttrList
  ClangDiagnosticFrontend
  ClangDiagnosticLex
  ClangDiagnosticSema
  ClangDeclNodes
  ClangStmtNodes)
