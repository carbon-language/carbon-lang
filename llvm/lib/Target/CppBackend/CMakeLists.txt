add_llvm_target(CppBackendCodeGen
  CPPBackend.cpp
  )

add_llvm_library_dependencies(LLVMCppBackendCodeGen
  LLVMCore
  LLVMCppBackendInfo
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
