add_llvm_library(LLVMPowerPCInfo
  PowerPCTargetInfo.cpp
  )

add_dependencies(LLVMPowerPCInfo PowerPCCommonTableGen)
