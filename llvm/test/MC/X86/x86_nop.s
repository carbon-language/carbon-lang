# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=geode %s -o %t
# RUN: llvm-objdump -disassemble %t | FileCheck %s

# CHECK-NOT: nopw
inc %eax
.align 8
inc %eax
