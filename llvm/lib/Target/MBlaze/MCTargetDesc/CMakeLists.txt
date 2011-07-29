add_llvm_library(LLVMMBlazeDesc
  MBlazeAsmBackend.cpp
  MBlazeMCAsmInfo.cpp
  MBlazeMCCodeEmitter.cpp
  MBlazeMCTargetDesc.cpp
  )

add_llvm_library_dependencies(LLVMMBlazeDesc
  LLVMMBlazeAsmPrinter
  LLVMMBlazeInfo
  LLVMMC
  LLVMSupport
  )

add_dependencies(LLVMMBlazeDesc MBlazeCommonTableGen)
