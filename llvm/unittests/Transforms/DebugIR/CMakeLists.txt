set(LLVM_LINK_COMPONENTS
  Instrumentation
  )

add_llvm_unittest(DebugIRTests
  DebugIR.cpp
  )
