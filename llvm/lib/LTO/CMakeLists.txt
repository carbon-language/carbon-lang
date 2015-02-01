add_llvm_library(LLVMLTO
  LTOModule.cpp
  LTOCodeGenerator.cpp
  )

add_dependencies(LLVMLTO intrinsics_gen)
