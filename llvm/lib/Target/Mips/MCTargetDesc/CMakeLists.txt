add_llvm_library(LLVMMipsDesc
  MipsMCTargetDesc.cpp
  MipsMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMMipsDesc
  LLVMMC
  LLVMMipsAsmPrinter
  LLVMMipsInfo
  LLVMSupport
  )

add_dependencies(LLVMMipsDesc MipsCommonTableGen)
