set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginInstructionARM64
  EmulateInstructionARM64.cpp
  )
