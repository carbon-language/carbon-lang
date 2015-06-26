
add_llvm_library(LLVMAMDGPUDesc
  AMDGPUAsmBackend.cpp
  AMDGPUELFObjectWriter.cpp
  AMDGPUMCCodeEmitter.cpp
  AMDGPUMCTargetDesc.cpp
  AMDGPUMCAsmInfo.cpp
  AMDGPUTargetStreamer.cpp
  R600MCCodeEmitter.cpp
  SIMCCodeEmitter.cpp
  )
