set(LLVM_LINK_COMPONENTS
  Support
  XRay
  TestingSupport
  )

add_llvm_unittest(XRayTests
  FDRBlockIndexerTest.cpp
  FDRBlockVerifierTest.cpp
  FDRProducerConsumerTest.cpp
  FDRRecordPrinterTest.cpp
  FDRRecordsTest.cpp
  FDRTraceWriterTest.cpp
  GraphTest.cpp
  ProfileTest.cpp
  )

add_dependencies(XRayTests intrinsics_gen)
