set(LLVM_NO_RTTI 1)

add_clang_library(clangBasic
  Diagnostic.cpp
  FileManager.cpp
  IdentifierTable.cpp
  LangOptions.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  TokenKinds.cpp
  )

add_dependencies(clangBasic 
                 ClangDiagnosticAnalysis
                 ClangDiagnosticAST
                 ClangDiagnosticCommon
                 ClangDiagnosticDriver
                 ClangDiagnosticFrontend
                 ClangDiagnosticLex
                 ClangDiagnosticParse
                 ClangDiagnosticSema)
