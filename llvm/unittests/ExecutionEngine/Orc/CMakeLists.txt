set(LLVM_LINK_COMPONENTS
  Core
  OrcJIT
  Support
  )

add_llvm_unittest(OrcJITTests
  IndirectionUtilsTest.cpp
  GlobalMappingLayerTest.cpp
  LazyEmittingLayerTest.cpp
  ObjectTransformLayerTest.cpp
  OrcTestCommon.cpp
  )
