set(LLVM_LINK_COMPONENTS
  MC
  MCAnalysis
  Support
  )

add_llvm_unittest(MCTests
  MCAtomTest.cpp
  StringTableBuilderTest.cpp
  YAMLTest.cpp
  )
