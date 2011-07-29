add_llvm_library(LLVMSparcDesc
  SparcMCTargetDesc.cpp
  SparcMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMSparcDesc
  LLVMMC
  LLVMSparcInfo
  LLVMSupport
  )

add_dependencies(LLVMSparcDesc SparcCommonTableGen)
