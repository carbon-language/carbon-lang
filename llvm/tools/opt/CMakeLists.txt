set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  BitWriter
  Core
  IPA
  IPO
  IRReader
  InstCombine
  Instrumentation
  MC
  ObjCARCOpts
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  )

# Support plugins.
set(LLVM_NO_DEAD_STRIP 1)

add_llvm_tool(opt
  AnalysisWrappers.cpp
  GraphPrinters.cpp
  NewPMDriver.cpp
  Passes.cpp
  PrintSCC.cpp
  opt.cpp
  )
set_target_properties(opt PROPERTIES ENABLE_EXPORTS 1)
