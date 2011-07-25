set(LLVM_TARGET_DEFINITIONS PIC16.td)

tablegen(PIC16.inc -gen-llvmc)

set(LLVM_USED_LIBS CompilerDriver)
set(LLVM_LINK_COMPONENTS support)

add_llvm_example(mcc16
  Hooks.cpp
  Main.cpp
  )
