set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontend
  AnalysisConsumer.cpp
  ASTConsumers.cpp
  Backend.cpp
  CacheTokens.cpp
  DeclXML.cpp
  DependencyFile.cpp
  DiagChecker.cpp
  DocumentXML.cpp
  FixItRewriter.cpp
  GeneratePCH.cpp
  HTMLDiagnostics.cpp
  HTMLPrint.cpp
  InitHeaderSearch.cpp
  InitPreprocessor.cpp
  ManagerRegistry.cpp
  PCHReader.cpp
  PCHReaderDecl.cpp
  PCHReaderStmt.cpp
  PCHWriter.cpp
  PCHWriterDecl.cpp
  PCHWriterStmt.cpp
  PlistDiagnostics.cpp
  PrintParserCallbacks.cpp
  PrintPreprocessedOutput.cpp
  RewriteBlocks.cpp
  RewriteMacros.cpp
  RewriteObjC.cpp
  RewriteTest.cpp
  StmtXML.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  TypeXML.cpp
  Warnings.cpp
  )

add_dependencies(clangSema ClangDiagnosticFrontend)
