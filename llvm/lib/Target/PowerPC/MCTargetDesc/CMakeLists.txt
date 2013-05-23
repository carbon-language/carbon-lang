add_llvm_library(LLVMPowerPCDesc
  PPCAsmBackend.cpp
  PPCMCTargetDesc.cpp
  PPCMCAsmInfo.cpp
  PPCMCCodeEmitter.cpp
  PPCMCExpr.cpp
  PPCPredicates.cpp
  PPCELFObjectWriter.cpp
  )

add_dependencies(LLVMPowerPCDesc PowerPCCommonTableGen)
