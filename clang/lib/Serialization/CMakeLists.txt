set(LLVM_LINK_COMPONENTS
  BitReader
  Support
  )


add_clang_library(clangSerialization
  ASTCommon.cpp
  ASTReader.cpp
  ASTReaderDecl.cpp
  ASTReaderStmt.cpp
  ASTWriter.cpp
  ASTWriterDecl.cpp
  ASTWriterStmt.cpp
  GeneratePCH.cpp
  GlobalModuleIndex.cpp
  Module.cpp
  ModuleManager.cpp

  ADDITIONAL_HEADERS
  ASTCommon.h
  ASTReaderInternals.h
  )

target_link_libraries(clangSerialization
  clangAST
  clangBasic
  clangLex
  clangSema
  )
