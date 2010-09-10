add_llvm_example(Kaleidoscope-Ch7
  toy.cpp
  )

add_llvm_link_components(Kaleidoscope-Ch7 jit interpreter native)
target_link_libraries(Kaleidoscope-Ch7 LLVMCore)
