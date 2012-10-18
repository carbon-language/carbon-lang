set(LLVM_LINK_COMPONENTS bitreader asmparser bitwriter instrumentation scalaropts ipo vectorize)

add_llvm_tool(opt
  AnalysisWrappers.cpp
  GraphPrinters.cpp
  PrintSCC.cpp
  opt.cpp
  )
