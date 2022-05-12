# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-nm %t.so | FileCheck --check-prefix=NM %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=RELOC %s

## R_PPC_ADDR32 is an absolute relocation type.
## In PIC mode, it creates a relative relocation if the symbol is non-preemptable.

# NM: 00030204 d b

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x30204 R_PPC_RELATIVE - 0x30204
# RELOC-NEXT:   0x30200 R_PPC_ADDR32 a 0
# RELOC-NEXT: }

.globl a, b
.hidden b

.data
.long a
b:
.long b
