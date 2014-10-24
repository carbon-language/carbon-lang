set(LLVM_LINK_COMPONENTS
  MC
  Support
  )

add_llvm_unittest(MCTests
  StringTableBuilderTest.cpp
  YAMLTest.cpp
  )
