set(LLVM_TARGET_DEFINITIONS Simple.td)

tablegen(Simple.inc -gen-llvmc)

set(LLVM_USED_LIBS CompilerDriver)
set(LLVM_LINK_COMPONENTS support)

add_llvm_example(Simple
  Simple.cpp
  )
