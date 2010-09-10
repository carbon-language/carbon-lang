add_llvm_example(HowToUseJIT
  HowToUseJIT.cpp
  )

add_llvm_link_components(HowToUseJIT jit interpreter nativecodegen)
