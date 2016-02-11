add_llvm_library(LLVMGlobalISel
        IRTranslator.cpp
        MachineIRBuilder.cpp
  )

add_dependencies(LLVMGlobalISel intrinsics_gen)
