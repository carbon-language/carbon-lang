add_llvm_tool(opt
  AnalysisWrappers.cpp
  GraphPrinters.cpp
  PrintSCC.cpp
  opt.cpp
  )

target_link_libraries(opt
  LLVMBitReader
  LLVMAsmParser
  LLVMBitWriter
  LLVMInstrumentation
  LLVMScalarOpts
  LLVMipo
  )
