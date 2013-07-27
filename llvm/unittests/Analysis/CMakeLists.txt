set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  )

add_llvm_unittest(AnalysisTests
  CFGTest.cpp
  ScalarEvolutionTest.cpp
  )
