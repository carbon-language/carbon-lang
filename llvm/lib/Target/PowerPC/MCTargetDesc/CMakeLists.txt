add_llvm_library(LLVMPowerPCDesc
  PPCAsmBackend.cpp
  PPCMCTargetDesc.cpp
  PPCMCAsmInfo.cpp
  PPCMCCodeEmitter.cpp
  PPCPredicates.cpp
  )

add_llvm_library_dependencies(LLVMPowerPCDesc
  LLVMMC
  LLVMPowerPCAsmPrinter
  LLVMPowerPCInfo
  LLVMSupport
  )

add_dependencies(LLVMPowerPCDesc PowerPCCommonTableGen)
