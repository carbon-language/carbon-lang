add_llvm_library(LLVMCellSPUDesc
  SPUMCTargetDesc.cpp
  SPUMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMCellSPUDesc
  LLVMCellSPUInfo
  LLVMMC
  )

add_dependencies(LLVMCellSPUDesc CellSPUCommonTableGen)
