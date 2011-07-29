add_llvm_target(CBackend
  CBackend.cpp
  )

add_llvm_library_dependencies(LLVMCBackend
  LLVMAnalysis
  LLVMCBackendInfo
  LLVMCodeGen
  LLVMCore
  LLVMMC
  LLVMScalarOpts
  LLVMSupport
  LLVMTarget
  LLVMTransformUtils
  )

add_subdirectory(TargetInfo)
