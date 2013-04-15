
add_llvm_library(LLVMR600Desc
  AMDGPUAsmBackend.cpp
  AMDGPUELFObjectWriter.cpp
  AMDGPUMCTargetDesc.cpp
  AMDGPUMCAsmInfo.cpp
  R600MCCodeEmitter.cpp
  SIMCCodeEmitter.cpp
  )

add_dependencies(LLVMR600Desc AMDGPUCommonTableGen)
