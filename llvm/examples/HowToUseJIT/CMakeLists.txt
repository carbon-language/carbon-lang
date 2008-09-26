set(LLVM_LINK_COMPONENTS jit interpreter nativecodegen)

add_llvm_executable(HowToUseJIT
  HowToUseJIT.cpp
  )
