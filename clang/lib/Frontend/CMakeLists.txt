set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontend
  ASTConsumers.cpp
  ASTUnit.cpp
  AnalysisConsumer.cpp
  Backend.cpp
  CacheTokens.cpp
  CompilerInstance.cpp
  DeclXML.cpp
  DependencyFile.cpp
  DiagChecker.cpp
  DocumentXML.cpp
  FixItRewriter.cpp
  FrontendOptions.cpp
  GeneratePCH.cpp
  HTMLDiagnostics.cpp
  HTMLPrint.cpp
  InitHeaderSearch.cpp
  InitPreprocessor.cpp
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
  VerifyDiagnosticsClient.cpp
  Warnings.cpp
  )

IF(MSVC)
  get_target_property(NON_ANSI_COMPILE_FLAGS clangFrontend COMPILE_FLAGS)
  string(REPLACE /Za
    "" NON_ANSI_COMPILE_FLAGS
    ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(clangFrontend PROPERTIES COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
ENDIF(MSVC)

add_dependencies(clangFrontend 
  ClangDiagnosticFrontend 
  ClangDiagnosticLex
  ClangDiagnosticSema)
