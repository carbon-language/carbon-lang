set(LLVM_LINK_COMPONENTS
  object
  )

add_llvm_unittest(ObjectTests
  YAMLTest.cpp
  )
