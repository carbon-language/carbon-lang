set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  CodeGen
  MIRParser
  Support
  Target
  )

add_llvm_unittest(MITests
  LiveIntervalTest.cpp
  )
