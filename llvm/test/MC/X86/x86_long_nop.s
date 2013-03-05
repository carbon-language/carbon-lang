# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i686 %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s

# Ensure alignment directives also emit sequences of 15-byte NOPs on processors
# capable of using long NOPs.
inc %eax
.align 32
inc %eax
# CHECK: 0:  inc
# CHECK-NEXT: 1:  nop
# CHECK-NEXT: 10:  nop
# CHECK-NEXT: 1f:  nop
# CHECK-NEXT: 20:  inc
