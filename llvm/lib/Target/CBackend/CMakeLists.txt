add_llvm_target(CBackendCodeGen
  CBackend.cpp
  )

add_llvm_library_dependencies(LLVMCBackendCodeGen
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
