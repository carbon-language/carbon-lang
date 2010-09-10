add_llvm_example(ModuleMaker
  ModuleMaker.cpp
  )

target_link_libraries(ModuleMaker LLVMBitWriter LLVMCore)
