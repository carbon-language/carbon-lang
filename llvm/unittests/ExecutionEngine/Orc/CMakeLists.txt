set(LLVM_LINK_COMPONENTS
  Core
  OrcJIT
  Support
  )

add_llvm_unittest(OrcJITTests
  IndirectionUtilsTest.cpp
  LazyEmittingLayerTest.cpp
  OrcTestCommon.cpp
  )
