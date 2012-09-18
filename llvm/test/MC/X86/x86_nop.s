# RUN: llvm-mc -filetype=obj -mcpu=geode %s -o %t
# RUN: llvm-objdump -disassemble %t | FileCheck %s

# CHECK-NOT: nopw
inc %eax
.align 8
inc %eax
