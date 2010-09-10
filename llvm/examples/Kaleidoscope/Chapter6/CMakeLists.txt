add_llvm_example(Kaleidoscope-Ch6
  toy.cpp
  )

add_llvm_link_components(Kaleidoscope-Ch6 jit interpreter native)
target_link_libraries(Kaleidoscope-Ch6 LLVMCore)
