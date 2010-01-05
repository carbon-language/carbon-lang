add_llvm_library(LLVMInstCombine
  InstructionCombining.cpp
  InstCombineCasts.cpp
  InstCombineCompares.cpp
  InstCombinePHI.cpp
  InstCombineSimplifyDemanded.cpp
  )

target_link_libraries (LLVMInstCombine LLVMTransformUtils)
