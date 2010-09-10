add_llvm_example(BrainF
  BrainF.cpp
  BrainFDriver.cpp
  )

add_llvm_link_components(BrainF jit nativecodegen interpreter)
target_link_libraries(BrainF LLVMBitWriter)
