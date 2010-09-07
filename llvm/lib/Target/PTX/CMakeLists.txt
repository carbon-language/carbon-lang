set(LLVM_TARGET_DEFINITIONS PTX.td)

add_llvm_target(PTXCodeGen
  )

target_link_libraries (LLVMPTXCodeGen LLVMSelectionDAG)
