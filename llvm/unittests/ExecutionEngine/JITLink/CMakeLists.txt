set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  JITLink
  MC
  Object
  RuntimeDyld
  Support
  )

add_llvm_unittest(JITLinkTests
    JITLinkTestCommon.cpp
    JITLinkTest_MachO_x86_64_Tests.cpp
  )

target_link_libraries(JITLinkTests PRIVATE LLVMTestingSupport)
