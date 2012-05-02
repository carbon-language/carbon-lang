set( LLVM_USED_LIBS
  clangAST
  clangBasic
  clangDriver
  clangEdit
  clangLex
  clangParse
  clangSema
  clangSerialization
  )

add_clang_library(clangFrontend
  ASTConsumers.cpp
  ASTMerge.cpp
  ASTUnit.cpp
  CacheTokens.cpp
  ChainedDiagnosticConsumer.cpp
  ChainedIncludesSource.cpp
  CompilerInstance.cpp
  CompilerInvocation.cpp
  CreateInvocationFromCommandLine.cpp
  DependencyFile.cpp
  DependencyGraph.cpp
  DiagnosticRenderer.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  HeaderIncludeGen.cpp
  InitHeaderSearch.cpp
  InitPreprocessor.cpp
  LangStandards.cpp
  LayoutOverrideSource.cpp
  LogDiagnosticPrinter.cpp
  MultiplexConsumer.cpp
  PrintPreprocessedOutput.cpp
  SerializedDiagnosticPrinter.cpp
  TextDiagnostic.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  VerifyDiagnosticConsumer.cpp
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
  ClangAttrClasses
  ClangAttrList
  ClangDiagnosticFrontend
  ClangDiagnosticLex
  ClangDiagnosticSema
  ClangDriverOptions
  ClangDeclNodes
  ClangStmtNodes)
