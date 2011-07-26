add_llvm_library(LLVMBlackfinDesc
  BlackfinMCTargetDesc.cpp
  BlackfinMCAsmInfo.cpp
  )
add_dependencies(LLVMBlackfinDesc BlackfinCommonTableGen)
