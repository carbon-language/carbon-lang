# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-nm %t.so | FileCheck --check-prefix=NM %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=RELOC %s

## R_RISCV_32 is an absolute relocation type.
## In PIC mode, it creates a relative relocation if the symbol is non-preemptable.

# NM: 000031fc d b

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x31FC R_RISCV_RELATIVE - 0x31FC
# RELOC-NEXT:   0x31F8 R_RISCV_32 a 0
# RELOC-NEXT: }

.globl a, b
.hidden b

.data
.long a
b:
.long b
