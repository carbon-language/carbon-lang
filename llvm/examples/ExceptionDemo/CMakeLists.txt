add_llvm_example(ExceptionDemo
  ExceptionDemo.cpp
  )

add_llvm_link_components(ExceptionDemo jit nativecodegen)
