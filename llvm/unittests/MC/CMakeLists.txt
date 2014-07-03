set(LLVM_LINK_COMPONENTS
  MCAnalysis
  )

add_llvm_unittest(MCTests
  MCAtomTest.cpp
  StringTableBuilderTest.cpp
  YAMLTest.cpp
  )
