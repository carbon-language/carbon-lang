# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -pie %t.o -o %tout
# RUN: llvm-objdump --section-headers %tout | FileCheck %s
# RUN: llvm-readobj --dynamic-table -r %tout | FileCheck %s --check-prefix=TAGS

## Check we produce DT_PLTREL/DT_JMPREL/DT_PLTGOT and DT_PLTRELSZ tags
## when there are no other relocations except R_*_IRELATIVE.

# CHECK:  Name          Size   VMA
# CHECK:  .rela.dyn   00000030 0000000000000248
# CHECK:  .got.plt    00000010 00000000000033b0

# TAGS:   Tag                Type                 Name/Value
# TAGS:   0x0000000000000007 RELA                 0x248
# TAGS:   0x0000000000000008 RELASZ               48 (bytes)
# TAGS:   0x0000000000000017 JMPREL               0x0
# TAGS:   0x0000000000000002 PLTRELSZ             0 (bytes)
# TAGS:   0x0000000000000003 PLTGOT               0x33B0
# TAGS:   0x0000000000000014 PLTREL               RELA

# TAGS:      Relocations [
# TAGS-NEXT:   Section {{.*}} .rela.dyn {
# TAGS-NEXT:     R_X86_64_IRELATIVE
# TAGS-NEXT:     R_X86_64_IRELATIVE
# TAGS-NEXT:   }
# TAGS-NEXT: ]

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 call foo
 call bar
