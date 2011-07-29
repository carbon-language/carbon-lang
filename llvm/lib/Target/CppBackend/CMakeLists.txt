add_llvm_target(CppBackend
  CPPBackend.cpp
  )

add_llvm_library_dependencies(LLVMCppBackend
  LLVMCore
  LLVMCppBackendInfo
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(TargetInfo)
