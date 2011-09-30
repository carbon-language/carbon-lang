add_llvm_library(LLVMMipsDesc
  MipsAsmBackend.cpp
  MipsMCAsmInfo.cpp
  MipsMCCodeEmitter.cpp
  MipsMCTargetDesc.cpp
  )

add_llvm_library_dependencies(LLVMMipsDesc
  LLVMMC
  LLVMMipsAsmPrinter
  LLVMMipsInfo
  LLVMSupport
  )

add_dependencies(LLVMMipsDesc MipsCommonTableGen)
