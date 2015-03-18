set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  AsmPrinter
  CodeGen
  Core
  IRReader
  MC
  ScalarOpts
  SelectionDAG
  Support
  )

# Support plugins.
set(LLVM_NO_DEAD_STRIP 1)

add_llvm_tool(llc
  llc.cpp
  )
export_executable_symbols(llc)
