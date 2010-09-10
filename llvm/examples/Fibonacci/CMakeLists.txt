add_llvm_example(Fibonacci
  fibonacci.cpp
  )

add_llvm_link_components(Fibonacci jit interpreter nativecodegen)
