set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginInstructionMIPS64
  EmulateInstructionMIPS64.cpp
  )
