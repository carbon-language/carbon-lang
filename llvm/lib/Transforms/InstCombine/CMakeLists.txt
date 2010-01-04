add_llvm_library(LLVMInstCombine
  InstructionCombining.cpp
  InstCombineSimplifyDemanded.cpp
  )

target_link_libraries (LLVMInstCombine LLVMTransformUtils)
