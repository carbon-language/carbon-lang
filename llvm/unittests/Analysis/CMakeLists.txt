set(LLVM_LINK_COMPONENTS
  IPA
  Analysis
  AsmParser
  Core
  Support
  )

add_llvm_unittest(AnalysisTests
  CallGraphTest.cpp
  CFGTest.cpp
  LazyCallGraphTest.cpp
  ScalarEvolutionTest.cpp
  MixedTBAATest.cpp
  )
