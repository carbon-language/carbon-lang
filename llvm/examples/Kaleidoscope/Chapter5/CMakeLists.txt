add_llvm_example(Kaleidoscope-Ch5
  toy.cpp
  )

add_llvm_link_components(Kaleidoscope-Ch5 jit interpreter native)
target_link_libraries(Kaleidoscope-Ch5 LLVMCore)
