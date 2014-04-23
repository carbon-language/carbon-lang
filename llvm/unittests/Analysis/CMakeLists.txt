set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  )

add_llvm_unittest(AnalysisTests
  CFGTest.cpp
  LazyCallGraphTest.cpp
  ScalarEvolutionTest.cpp
  )
