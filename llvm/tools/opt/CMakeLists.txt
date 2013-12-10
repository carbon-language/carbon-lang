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

add_llvm_tool(opt
  AnalysisWrappers.cpp
  GraphPrinters.cpp
  PrintSCC.cpp
  opt.cpp
  )
set_target_properties(opt PROPERTIES ENABLE_EXPORTS 1)
