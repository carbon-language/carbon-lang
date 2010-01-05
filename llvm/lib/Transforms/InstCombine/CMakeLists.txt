add_llvm_library(LLVMInstCombine
  InstructionCombining.cpp
  InstCombineCasts.cpp
  InstCombineCompares.cpp
  InstCombineLoadStoreAlloca.cpp
  InstCombineMulDivRem.cpp
  InstCombinePHI.cpp
  InstCombineSelect.cpp
  InstCombineSimplifyDemanded.cpp
  InstCombineVectorOps.cpp
  )

target_link_libraries (LLVMInstCombine LLVMTransformUtils)
