set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_unittest(ObjectTests
  StringTableBuilderTest.cpp
  YAMLTest.cpp
  )
