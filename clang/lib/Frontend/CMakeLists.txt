set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontend
  ASTConsumers.cpp
  Backend.cpp
  FixItRewriter.cpp
  GeneratePCH.cpp
  HTMLDiagnostics.cpp
  HTMLPrint.cpp
  InitHeaderSearch.cpp
  InitPreprocessor.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  PCHReader.cpp
  PCHReaderDecl.cpp
  PCHReaderStmt.cpp
  PCHWriter.cpp
  PCHWriterDecl.cpp
  PCHWriterStmt.cpp
  PlistDiagnostics.cpp
  RewriteBlocks.cpp
  RewriteObjC.cpp
  ManagerRegistry.cpp
  )
