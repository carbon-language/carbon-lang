set(LLVM_LINK_COMPONENTS jit nativecodegen)
set(LLVM_REQUIRES_EH 1)

add_llvm_example(ExceptionDemo
  ExceptionDemo.cpp
  )

set_target_properties(ExceptionDemo PROPERTIES ENABLE_EXPORTS 1)
