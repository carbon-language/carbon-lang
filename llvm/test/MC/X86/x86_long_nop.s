# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-pc-linux-gnu %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=x86_64-apple-darwin10.0 %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-apple-darwin8 %s | llvm-objdump -d -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -triple=i686-pc-linux-gnu -mcpu=slm %s | llvm-objdump -d -no-show-raw-insn - | FileCheck --check-prefix=SLM %s

# Ensure alignment directives also emit sequences of 15-byte NOPs on processors
# capable of using long NOPs.
inc %eax
.p2align 5
inc %eax
# CHECK: 0:  inc
# CHECK-NEXT: 1:  nop
# CHECK-NEXT: 10:  nop
# CHECK-NEXT: 1f:  nop
# CHECK-NEXT: 20:  inc

# On Silvermont we emit only 7 byte NOPs since longer NOPs are not profitable
# SLM: 0:  inc
# SLM-NEXT: 1:  nop
# SLM-NEXT: 8:  nop
# SLM-NEXT: f:  nop
# SLM-NEXT: 16:  nop
# SLM-NEXT: 1d:  nop
# SLM-NEXT: 20:  inc
