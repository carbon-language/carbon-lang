set(LLVM_LINK_COMPONENTS
  Core
  Instrumentation
  Support
  )

add_llvm_unittest(DebugIRTests
  DebugIR.cpp
  )
