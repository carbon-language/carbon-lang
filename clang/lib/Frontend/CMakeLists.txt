set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontend
  FixItRewriter.cpp
  HTMLDiagnostics.cpp
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
  ManagerRegistry.cpp
  )
