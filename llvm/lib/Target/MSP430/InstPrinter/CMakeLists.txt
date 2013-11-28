add_llvm_library(LLVMMSP430AsmPrinter
  MSP430InstPrinter.cpp
  )

add_dependencies(LLVMMSP430AsmPrinter MSP430CommonTableGen)
