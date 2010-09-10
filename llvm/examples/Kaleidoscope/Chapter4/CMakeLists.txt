add_llvm_example(Kaleidoscope-Ch4
  toy.cpp
  )

add_llvm_link_components(Kaleidoscope-Ch4 jit interpreter native)
target_link_libraries(Kaleidoscope-Ch4 LLVMCore)
