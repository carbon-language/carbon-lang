set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangCodeGen
  clangAnalysis
  clangRewrite
  clangSema
  clangFrontend
  clangAST
  clangParse
  clangLex
  clangBasic
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  bitreader
  bitwriter
  codegen
  ipo
  selectiondag
  )

add_clang_executable(clang-cc
  AnalysisConsumer.cpp
  ASTConsumers.cpp
  Backend.cpp
  CacheTokens.cpp
  clang-cc.cpp
  DependencyFile.cpp
  DiagChecker.cpp
  GeneratePCH.cpp
  HTMLPrint.cpp
  PrintParserCallbacks.cpp
  PrintPreprocessedOutput.cpp
  RewriteBlocks.cpp
  RewriteMacros.cpp
  RewriteObjC.cpp
  RewriteTest.cpp
  Warnings.cpp
  )
