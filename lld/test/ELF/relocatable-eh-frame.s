# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -r %t.o %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.eh_frame {
# CHECK-NEXT:     0x20 R_X86_64_PC32 .foo 0x0
# CHECK-NEXT:     0x0 R_X86_64_NONE - 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.section .foo,"aG",@progbits,bar,comdat
.cfi_startproc
.cfi_endproc
