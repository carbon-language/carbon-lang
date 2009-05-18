set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangFrontend
  clangCodeGen
  clangAnalysis
  clangRewrite
  clangSema
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
  CacheTokens.cpp
  clang-cc.cpp
  DependencyFile.cpp
  DiagChecker.cpp
  PrintParserCallbacks.cpp
  PrintPreprocessedOutput.cpp
  RewriteMacros.cpp
  RewriteTest.cpp
  Warnings.cpp
  )
