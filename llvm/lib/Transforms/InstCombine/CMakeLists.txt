add_llvm_library(LLVMInstCombine
  InstructionCombining.cpp
  )

target_link_libraries (LLVMInstCombine LLVMTransformUtils)
