add_llvm_library(LLVMBlackfinDesc
  BlackfinMCTargetDesc.cpp
  BlackfinMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMBlackfinDesc
  LLVMBlackfinInfo
  LLVMMC
  )

add_dependencies(LLVMBlackfinDesc BlackfinCommonTableGen)
